# ML Project

Small housing price ML project — data cleaning, feature engineering, and model training.

Files of interest
- `main.py` — primary entrypoint to run training/evaluation

Quick start
1. Create a virtual environment: `python -m venv .venv`
2. Activate it and install dependencies: `pip install -r requirements.txt`
3. Put the rows you want predictions for into `input.csv` (same columns as training data, omit `median_house_value`).
4. Run training/inference: `python main.py` — predictions will be saved to `output.csv`.

Notes
- This repository currently contains CSV sample data; consider moving large datasets to a `data/` folder and adding them to `.gitignore` if they should not be committed.
