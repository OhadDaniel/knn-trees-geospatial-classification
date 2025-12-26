# Geospatial Classification with kNN, Trees, and Boosting

Predicting a city’s U.S. state from geographic coordinates (longitude, latitude) and visualizing how different models partition the map.

## Overview
This repository contains multiple classical machine learning approaches for geospatial multi-class classification:
- **kNN (FAISS)** for fast nearest-neighbor inference in 2D coordinate space
- **kNN-based anomaly detection** using nearest-neighbor distance statistics
- **Decision Trees** with depth/leaf constraints and decision-region visualization
- **Random Forest** and **XGBoost** as stronger tree-based baselines

## What I built
- **FAISS-based kNN classifier** with configurable `k` and distance metric (L1/L2)
- **Anomaly detection** based on 5-NN distance sums (ranking most unusual points)
- **Decision tree experiments** including restricted-depth variants and visualization utilities
- **Random Forest** and **XGBoost** experiments for comparison and smoother decision boundaries
- A **technical report** summarizing methodology and results

## Repository contents
- `knn.py` / `run_knn.py` / `plot_knn.py` — kNN training + evaluation + plots
- `anomaly_detection.py` — anomaly scoring with kNN distances
- `decision_trees.py` / `restricted_depth_tree.py` — tree experiments
- `random_forest.py` — Random Forest experiments
- `xgboost_experiment.py` — XGBoost experiments
- `visualize_tree.py` / `visualize_tree_50.py` — decision-region visualizations
- `helpers.py` — shared utilities
- `docs/technical_report.pdf` — detailed write-up and results

## Data
This repository does **not** include the dataset files.
Place the following files in the project root (or update paths inside the scripts):
- `train.csv`
- `validation.csv`
- `test.csv`
- `AD_test.csv`

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
