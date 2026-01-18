# Street Light Scheduling (ANN + EA + Fuzzy)

This project generates a synthetic street-light dataset, produces exploratory visualizations, trains an ANN to predict hourly activity, and optimizes a 24-hour lighting schedule using evolutionary algorithms (standard EA, fuzzy EA, and tuned fuzzy EA).

## Files
- **data_gen.py** — dataset generation → `street_light_dataset.csv`
- **plots.py** — creates 12 plots (`01_*.png` … `12_*.png`) + PCA summary prints
- **ann_model.py** — ANN pipeline (preprocessing, optimizer comparison, final training, evaluation plots)
- **ea_optimizer.py** — schedule optimization + convergence/analysis plots
- **main.py** — runs the full pipeline end-to-end

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
