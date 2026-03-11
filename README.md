# Optimizing OpenFOAM's interfacial momentum transports via Bayesian optimization

This project extends from JanisGeise's repository, adding a Bayesian optimisation workflow for interfacial momentum transport models in OpenFOAM wall boiling tutorial case. 

JanisGeise's repo: https://github.com/JanisGeise/BayesOpt_solverSettings

## Getting started - local execution

### Dependencies

The instructions and tests are tailored to:
- OpenFOAM-v2406 (JanisGeise's test cases)
- OpenFOAM-10 (wall boiling case)
- Python 3.11

### Files added
-Bayesian_OpenFOAM_tutorial/test_cases/wallBoiling_template/ #OpenFOAM case template + validation data
-Bayesian_OpenFOAM_tutorial/wallBoiling_local.yaml
-Bayesian_OpenFOAM_tutorial/run_wallBoiling.py
-Bayesian_OpenFOAM_tutorial/execution_wallBoiling.py
-Bayesian_OpenFOAM_tutorial/eval_wallBoiling_bo.py

### Set up before running

```bash
git clone https://github.com/zw114/BayesOpt_OpenFOAM_tutorial.git
cd BayesOpt_OpenFOAM_tutorial # repository top-level
python -m venv bopt
source bopt/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


### Quick run

```
source bopt/bin/activate # enactivate virtual environment
python run_wallBoiling.py wallBoiling_local.yaml &> log.wallBoiling
python eval_wallBoiling_bo.py #postProcessing
```
### Outputs

-run/wallBoiling_bo/
-output.wallBoiling_bo/
