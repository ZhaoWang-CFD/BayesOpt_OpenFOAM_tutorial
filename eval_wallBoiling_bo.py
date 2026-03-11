#!/usr/bin/env python3
# eval_bo_report.py

#!/usr/bin/env python3
# eval_wallBoiling_bo.py
#
# Features:
# - trials_by_index.csv / trials_by_score.csv
# - best_params.txt (best observed trial)
# - objective_bestsofar.png (stars for trials, line for best-so-far)
# - scatter_param_<param>.png for each parameter
# - surrogate_slice_<param>.png (optional, uses scikit-learn GP)
# - feature_importance.png/.csv (optional, surrogate sensitivity)
# - cv_pred_vs_obs.png/.csv (optional, leave-one-out CV)

import os
import glob
import argparse
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ax.service.ax_client import AxClient

# --- optional surrogate stack (sklearn) ---
HAS_SKLEARN = True
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
    from sklearn.model_selection import LeaveOneOut
except Exception:
    HAS_SKLEARN = False


def find_ax_client_json(exp_path: str) -> str:
    """Find Ax client json under exp_path."""
    cands = []
    cands += glob.glob(os.path.join(exp_path, "ax_client_int_*.json"))
    cands += glob.glob(os.path.join(exp_path, "ax-client_int_*.json"))
    if not cands:
        raise FileNotFoundError(
            f"Cannot find ax_client_int_*.json or ax-client_int_*.json under: {exp_path}"
        )
    return max(cands, key=os.path.getmtime)


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def save_objective_bestsofar(df_by_idx: pd.DataFrame, metric: str, out_png: str):
    x = df_by_idx["trial_index"].to_numpy(int)
    y = df_by_idx[metric].to_numpy(float)
    best = np.minimum.accumulate(y)

    plt.figure()
    # trials: stars (no line)
    plt.plot(x, y, linestyle="None", marker="*", label=f"{metric} (each trial)")
    # best-so-far: continuous line
    plt.plot(x, best, linestyle="-", marker="o", label="best-so-far (min up to trial)")
    plt.xlabel("trial_index")
    plt.ylabel(f"{metric} (lower is better)")
    plt.title("Objective and best-so-far")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_param_scatter(df: pd.DataFrame, metric: str, params: list[str], out_dir: str):
    for p in params:
        if p not in df.columns:
            continue
        plt.figure()
        plt.scatter(df[p].to_numpy(float), df[metric].to_numpy(float))
        plt.xlabel(p)
        plt.ylabel(metric)
        plt.title(f"{metric} vs {p}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"scatter_param_{p}.png"), dpi=200)
        plt.close()


# ------------------ surrogate (sklearn GP) ------------------

def _bounds_from_ax_or_data(axc: AxClient, df: pd.DataFrame, params: list[str]) -> dict:
    """Try bounds from Ax search space; fallback to observed min/max."""
    bounds = {}
    for p in params:
        lo = None
        hi = None
        try:
            sp = axc.experiment.search_space.parameters[p]
            lo, hi = float(sp.lower), float(sp.upper)
        except Exception:
            lo, hi = float(df[p].min()), float(df[p].max())
        if hi == lo:
            hi = lo + 1.0
        bounds[p] = (lo, hi)
    return bounds


def _scale_to_01(X: np.ndarray, bounds: dict, params: list[str]) -> np.ndarray:
    Xs = np.empty_like(X, dtype=float)
    for j, p in enumerate(params):
        lo, hi = bounds[p]
        Xs[:, j] = (X[:, j] - lo) / (hi - lo)
    return Xs


def fit_surrogate_gp(axc: AxClient, df: pd.DataFrame, metric: str, params: list[str]):
    """Fit GP surrogate y = f(params). Returns (gp, bounds)."""
    bounds = _bounds_from_ax_or_data(axc, df, params)

    X = df[params].to_numpy(dtype=float)
    y = df[metric].to_numpy(dtype=float)
    Xs = _scale_to_01(X, bounds, params)

    d = Xs.shape[1]
    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=np.ones(d),
        length_scale_bounds=(1e-2, 1e2)
    ) + WhiteKernel(
        noise_level=1e-6,
        noise_level_bounds=(1e-10, 1e-1)
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=0
    )
    gp.fit(Xs, y)
    return gp, bounds


def predict_1d_slice(gp, bounds: dict, params: list[str], base_params: dict, p_name: str, n: int = 60):
    """Vary one parameter across its bounds; keep others fixed. Return xs_raw, mu, sigma."""
    lo, hi = bounds[p_name]
    xs = np.linspace(lo, hi, n)

    Xraw = np.zeros((n, len(params)), dtype=float)
    for i, v in enumerate(xs):
        for j, p in enumerate(params):
            Xraw[i, j] = float(v) if p == p_name else float(base_params[p])

    Xs = _scale_to_01(Xraw, bounds, params)
    mu, sigma = gp.predict(Xs, return_std=True)
    return xs, mu.astype(float), sigma.astype(float)


def save_surrogate_slices(gp, bounds: dict, metric: str, out_dir: str, params: list[str], best_params: dict):
    for p in params:
        xs, mu, sig = predict_1d_slice(gp, bounds, params, best_params, p, n=60)
        plt.figure()
        plt.plot(xs, mu, label="surrogate mean")
        plt.fill_between(xs, mu - 2 * sig, mu + 2 * sig, alpha=0.2, label="±2σ")
        plt.xlabel(p)
        plt.ylabel(metric)
        plt.title(f"Surrogate slice: {metric} vs {p} (others fixed at best)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"surrogate_slice_{p}.png"), dpi=200)
        plt.close()


def save_feature_importance(gp, bounds: dict, metric: str, out_dir: str, params: list[str], best_params: dict):
    """
    Robust importance for demo:
    for each param, vary it across bounds and measure how much surrogate mean changes (range).
    Normalize to sum=1.
    """
    scores = {}
    for p in params:
        xs, mu, _ = predict_1d_slice(gp, bounds, params, best_params, p, n=80)
        scores[p] = float(np.max(mu) - np.min(mu))

    total = sum(scores.values())
    if total <= 0:
        total = 1.0
    imp = {k: v / total for k, v in scores.items()}
    ser = pd.Series(imp).sort_values(ascending=False)

    ser.to_csv(os.path.join(out_dir, "feature_importance.csv"), header=["importance"])

    plt.figure()
    ser.plot(kind="bar")
    plt.ylabel("importance (normalized)")
    plt.title("Feature importance (surrogate sensitivity)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=200)
    plt.close()


def save_cross_validation(df: pd.DataFrame, metric: str, params: list[str], out_dir: str):
    """Leave-one-out CV with GP to show predicted vs observed."""
    X = df[params].to_numpy(dtype=float)
    y = df[metric].to_numpy(dtype=float)

    loo = LeaveOneOut()
    obs = []
    pred = []

    for train_idx, test_idx in loo.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        lo = Xtr.min(axis=0)
        hi = Xtr.max(axis=0)
        hi = np.where(hi == lo, lo + 1.0, hi)

        Xtr_s = (Xtr - lo) / (hi - lo)
        Xte_s = (Xte - lo) / (hi - lo)

        d = Xtr_s.shape[1]
        kernel = C(1.0, (1e-3, 1e3)) * RBF(
            length_scale=np.ones(d),
            length_scale_bounds=(1e-2, 1e2)
        ) + WhiteKernel(
            noise_level=1e-6,
            noise_level_bounds=(1e-10, 1e-1)
        )

        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=1, random_state=0)
        gp.fit(Xtr_s, ytr)
        yhat = float(gp.predict(Xte_s)[0])

        obs.append(float(yte[0]))
        pred.append(yhat)

    out = pd.DataFrame({"observed": obs, "predicted": pred})
    out.to_csv(os.path.join(out_dir, "cv_pred_vs_obs.csv"), index=False)

    plt.figure()
    plt.scatter(obs, pred)
    lo2 = min(min(obs), min(pred))
    hi2 = max(max(obs), max(pred))
    plt.plot([lo2, hi2], [lo2, hi2])
    plt.xlabel("observed")
    plt.ylabel("predicted")
    plt.title(f"LOO-CV: predicted vs observed ({metric})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_pred_vs_obs.png"), dpi=200)
    plt.close()


# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="run/wallBoiling_bo", help="Experiment folder (default: run/wallBoiling_bo)")
    ap.add_argument("--out", default="output/wallBoiling_bo", help="Output folder (default: output/wallBoiling_bo)")
    ap.add_argument("--metric", default="weighted_mse", help="Objective metric column (default: weighted_mse)")
    args = ap.parse_args()

    exp_path = args.exp
    out_dir = args.out
    metric = args.metric
    safe_makedirs(out_dir)

    ax_file = find_ax_client_json(exp_path)
    axc = AxClient().load_from_json_file(ax_file)

    df = axc.get_trials_data_frame()
    df = df[df["trial_status"] == "COMPLETED"].copy()
    if len(df) == 0:
        raise RuntimeError("No COMPLETED trials found.")

    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found. Columns: {list(df.columns)}")

    ignore = {"trial_index", "arm_name", "trial_status", "generation_node", metric}
    param_cols = [c for c in df.columns if c not in ignore]

    # tables
    df_by_idx = df.sort_values("trial_index")
    df_by_score = df.sort_values(metric)

    df_by_idx.to_csv(os.path.join(out_dir, "trials_by_index.csv"), index=False)
    df_by_score.to_csv(os.path.join(out_dir, "trials_by_score.csv"), index=False)

    # best observed trial (most robust for demo)
    best_row = df_by_score.iloc[0]
    best_params_obs = {p: float(best_row[p]) for p in param_cols}

    with open(os.path.join(out_dir, "best_params.txt"), "w", encoding="utf-8") as f:
        f.write(f"ax_file: {ax_file}\n")
        f.write(f"metric: {metric}\n\n")
        f.write(f"best_trial_index (observed): {int(best_row['trial_index'])}\n")
        f.write(f"best_{metric} (observed): {float(best_row[metric])}\n\n")
        f.write("best_params (observed):\n")
        for k, v in best_params_obs.items():
            f.write(f"  {k}: {v}\n")

    # plots
    save_objective_bestsofar(df_by_idx, metric, os.path.join(out_dir, "objective_bestsofar.png"))
    save_param_scatter(df, metric, param_cols, out_dir)

    # surrogate-based analysis (optional)
    # remove old error marker
    err_path = os.path.join(out_dir, "surrogate_errors.txt")
    if os.path.exists(err_path):
        os.remove(err_path)

    if HAS_SKLEARN:
        try:
            gp, bounds = fit_surrogate_gp(axc, df, metric, param_cols)
            save_surrogate_slices(gp, bounds, metric, out_dir, param_cols, best_params_obs)
            save_feature_importance(gp, bounds, metric, out_dir, param_cols, best_params_obs)
            save_cross_validation(df, metric, param_cols, out_dir)
        except Exception:
            with open(err_path, "w") as f:
                f.write(traceback.format_exc())
    else:
        with open(err_path, "w") as f:
            f.write("scikit-learn not available; skipped surrogate/CV/importance.\n")

    print("Saved outputs to:", out_dir)


if __name__ == "__main__":
    main()
