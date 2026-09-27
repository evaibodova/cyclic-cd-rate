# Reproducing the experiments

Run these commands from this directory with Python 3.12.5. The reported runs used macOS 15.6.1.

```sh
python -m pip install -r requirements.txt
python -c "from report import run_experiments; run_experiments(show=False)"
python -c "from permutations import launch_experiment_with_report; launch_experiment_with_report(show=False)"
python paper_plots.py
```

The six datasets are downloaded from the official LIBSVM collection on first use and cached in `data/`. Tables and figures are written to `results/`. The full runs can take considerable time.
