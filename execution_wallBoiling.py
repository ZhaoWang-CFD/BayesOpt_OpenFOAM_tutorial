"""Run parameter variations of base simulation to evaluate trial.
"""

from os import listdir
from os.path import isdir, join
from typing import Dict, Union
from copy import deepcopy
from collections import defaultdict
from math import sqrt
from time import sleep
from smartsim import Experiment
from smartsim.settings import RunSettings, MpirunSettings, SrunSettings
from smartsim.settings.base import BatchSettings
from smartsim.entity import Model
import numpy as np
from pandas import read_csv
import glob


LOG10_KEYS = ("tolerance", "relTol", "toleranceCoarsest", "relTolCoarsest")


def is_float(value: str) -> bool:
    try:
        _ = float(value)
        return True
    except:
        return False


def find_closest_time(path: str, time: Union[float, int]) -> str:
    dirs = [f for f in listdir(path) if isdir(join(path, f)) and is_float(f)]
    dirs_num = np.array([float(f) for f in dirs])
    closest = np.argmin(np.absolute(dirs_num - time))
    return dirs[closest]

def find_latest_time(path: str) -> str:
    """Return the directory name with the largest numeric value (e.g. '4' or '4.5')."""
    dirs = [f for f in listdir(path) if isdir(join(path, f)) and is_float(f)]
    if not dirs:
        raise FileNotFoundError(f"No time directories found in: {path}")
    dirs_num = np.array([float(f) for f in dirs])
    latest = int(np.argmax(dirs_num))
    return dirs[latest]


def load_xy(filepath: str):
    """Read a whitespace-separated xy file.
    Column 1 is r (or distance), column 2..N are values."""
    arr = np.genfromtxt(filepath, comments="#")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = arr[~np.isnan(arr).any(axis=1)]
    if arr.shape[1] < 2:
        raise ValueError(f"File has <2 columns: {filepath}")
    r = arr[:, 0].astype(float)
    data = arr[:, 1:].astype(float)
    idx = np.argsort(r)
    return r[idx], data[idx]


from os.path import isabs, abspath, dirname

def load_exp_csv(filepath: str):
    """Experimental data file: whitespace separated 2 columns: r value.
    Accept .txt or .csv. Path is resolved relative to this repo."""
    # make path robust: relative to execution.py location
    if not isabs(filepath):
        filepath = join(dirname(abspath(__file__)), filepath)

    # whitespace-separated, no header
    df = read_csv(filepath, header=None, sep=r"\s+", comment="#")
    r = df.iloc[:, 0].to_numpy(dtype=float)
    v = df.iloc[:, 1].to_numpy(dtype=float)
    idx = np.argsort(r)
    return r[idx], v[idx]


def weighted_mse_from_trial(trial_dir: str, config: dict) -> float:
    """Objective for one trial:
    use postProcess/radialRaw/<lastTime>/radial.xy and compare to exp CSV."""
    obj_cfg = config["objective"]
    w_alpha = float(obj_cfg.get("w_alpha", 0.5))
    w_d = float(obj_cfg.get("w_d", 0.5))

    exp_alpha_csv = obj_cfg["exp_alpha_csv"]
    exp_d_csv = obj_cfg["exp_d_csv"]

    sim_root = obj_cfg.get("sim_root", "postProcessing/radialRaw")
    sim_file = obj_cfg.get("sim_file", "radial.xy")

    # Column numbers in the ORIGINAL file, starting from 1.
    # Example: columns are [1=r, 2=alpha.gas, 3=d.gas] -> alpha_col=2, d_col=3
    alpha_col = int(obj_cfg["alpha_col"])
    d_col = int(obj_cfg["d_col"])

    sim_time_root = join(trial_dir, sim_root)
    t_last = find_latest_time(sim_time_root)
    sim_path = join(sim_time_root, t_last, sim_file)

    r_sim, data = load_xy(sim_path)

    ncols_total = 1 + data.shape[1]  # including r
    if alpha_col < 2 or alpha_col > ncols_total:
        raise ValueError(f"alpha_col={alpha_col} out of range for {sim_path} with {ncols_total} cols")
    if d_col < 2 or d_col > ncols_total:
        raise ValueError(f"d_col={d_col} out of range for {sim_path} with {ncols_total} cols")

    alpha_sim = data[:, alpha_col - 2]  # -2 because data excludes r and alpha_col is 1-based
    d_sim = data[:, d_col - 2]

    r_exp_a, a_exp = load_exp_csv(exp_alpha_csv)
    r_exp_d, d_exp = load_exp_csv(exp_d_csv)

    a_sim_i = np.interp(r_exp_a, r_sim, alpha_sim, left=alpha_sim[0], right=alpha_sim[-1])
    d_sim_i = np.interp(r_exp_d, r_sim, d_sim, left=d_sim[0], right=d_sim[-1])

    scale_a = float(np.std(a_exp)) if float(np.std(a_exp)) > 0 else 1.0
    scale_d = float(np.std(d_exp)) if float(np.std(d_exp)) > 0 else 1.0

    mse_a = float(np.mean(((a_sim_i - a_exp) / scale_a) ** 2))
    mse_d = float(np.mean(((d_sim_i - d_exp) / scale_d) ** 2))

    return w_alpha * mse_a + w_d * mse_d


def batch_settings_from_config(exp: Experiment, batch_config: dict) -> Union[BatchSettings, None]:
    if batch_config is not None:
        batch_args = batch_config.get("batch_args")
        nodes = batch_args.get("nodes") if batch_args else None
        bs = exp.create_batch_settings(batch_args=batch_args, nodes=nodes)
        if "preamble" in batch_config:
            bs.add_preamble(batch_config["preamble"])
    else:
        bs = None
    return bs


def run_parameter_variation(
    exp: Experiment, trials: dict, config: dict, time_idx: int
) -> Dict[int, float]:
    # create a copy for each trial and link the processor folders
    opt_config = config["optimization"]
    param_group = config.get("param_group", "gamg")
    rs = RunSettings(exe="bash", exe_args="link_procs")
    bs = batch_settings_from_config(exp, config.get("batch_settings"))
    path = join(exp.exp_path, "base_sim", "processor0")
    startTime = find_closest_time(path, opt_config["startTime"][time_idx])
    endTime = float(startTime) + opt_config["duration"]
    sim_params = {
        "startTime": float(startTime),
        "endTime": endTime,
        "writeInterval": opt_config["writeInterval"],
        "deltaT" : opt_config["deltaT"],
        "baseCase": "../../base_sim",
    }
    gamg_params = {}
    for key in trials.keys():
        default = deepcopy(config["simulation"][param_group])
        for key_i, val_i in trials[key].items():
            if key_i in LOG10_KEYS:
                default[key_i] = 10**val_i
            else:
                default[key_i] = val_i
        gamg_params[key] = default
    params_full = [sim_params | gamg_params[key] for key in gamg_params.keys()]
    params = defaultdict(list)
    for d in params_full:
        for key, val in d.items():
            params[key].extend([val] * opt_config["n_repeat_trials"])
    keys_str = [str(key) for key in trials.keys()]
    ens = exp.create_ensemble(
        name=f"int_{time_idx}_trial_{'_'.join(keys_str)}",
        params=params,
        perm_strategy="step",
        run_settings=rs,
        batch_settings=None
    )
    base_case_path = config["simulation"]["base_case"]
    ens.attach_generator_files(to_configure=base_case_path)
    exp.generate(ens, overwrite=True, tag="!")
    exp.start(ens, block=True, summary=True)

    # run solver
    launcher = config["experiment"]["launcher"]
    solver = config["simulation"]["solver"]
    settings_class = MpirunSettings if launcher == "local" else SrunSettings
    if opt_config["repeated_trials_parallel"]:
        solver_models = []
        for model_i in ens.models:
            solver_settings = settings_class(
                exe=solver,
                exe_args=f"-case {model_i.path} -parallel",
                run_args=config["simulation"].get("run_args")
            )
            solver_models.append(
                exp.create_model(
                    name=f"{model_i.name}_{solver}",
                    run_settings=solver_settings,
                    batch_settings=bs
                )
            )
            exp.start(solver_models[-1], block=False)
        while not all(exp.finished(model_i) for model_i in solver_models):
            sleep(2)
    else:
        n_parallel = opt_config["batch_size"]
        for i in range(0, len(ens.models), n_parallel):
            ens_batch = ens.models[i:i+n_parallel]
            solver_models = []
            for model_i in ens_batch:
                solver_settings = settings_class(
                    exe=solver,
                    exe_args=f"-case {model_i.path} -parallel",
                    run_args=config["simulation"].get("run_args")
                )
                solver_models.append(
                    exp.create_model(
                        name=f"{model_i.name}_{solver}",
                        run_settings=solver_settings,
                        batch_settings=bs
                    )
                )
                exp.start(solver_models[-1], block=False)
            while not all(exp.finished(model_i) for model_i in solver_models):
                sleep(2)
        # compute objective value for each trial (your objective is in config["objective"])
    objectives = []
    for model in ens.models:
        try:
            j = weighted_mse_from_trial(model.path, config)
        except Exception:
            j = opt_config["bad_value"]
        objectives.append(j)
    nr = opt_config["n_repeat_trials"]
    if nr > 1:
        stats = [
            (np.mean(objectives[i:i+nr]), np.std(objectives[i:i+nr]) / sqrt(nr))
            for i in range(0, len(objectives), nr)
        ]
        obj = {
            key : stat_i for key, stat_i in zip(trials.keys(), stats)
        }
    else:
        obj = {
            key : (t_i, 0.0) for key, t_i in zip(trials.keys(), objectives)
        }
    return obj
