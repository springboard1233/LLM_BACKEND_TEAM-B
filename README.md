<div style="background:#0b1226;padding:18px;border-radius:12px;color:#fff;">
  <h1 style="margin:0;font-size:28px;">BFSI Predictive Modeling</h1>
  <p style="margin:6px 0 0;color:#cbd5e1;">Fraud detection pipeline — exploratory data analysis (EDA), preprocessing, modeling and evaluation (work in progress).</p>
</div>

<!-- Badges -->
<br>

<p>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="status" src="https://img.shields.io/badge/status-in--progress-yellow">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
</p>

## 🔎 Project Summary

This repository contains the machine learning backend work for a BFSI (Banking, Financial Services & Insurance) fraud-prediction project. The work currently focuses on data exploration and preprocessing of transactional data, preparing clean datasets for model training and evaluation.

> **Current state:** EDA and preprocessing implemented. Modeling/training and CI/CD pipelines are next steps.

## 📁 Folder Structure (relevant)

```
backend/ml/
├─ data/                      # Raw and processed CSV files
├─ eda_outputs/               # PNGs and CSV summaries produced by EDA
├─ models/                    # Saved model artifacts (trained models)
├─ notebook/                  # Analysis notebooks
├─ __init__.py
├─ evaluation.py              # Evaluation utilities (WIP)
├─ preprocessing.py           # Data cleaning & preprocessing script
├─ training.py                # Training script (WIP)
└─ README.md                  # This file
```

## ✅ What we have done

* Standardized pipeline to clean and preprocess `transactions.csv`.
* Implemented feature engineering (timestamp features, amount bins, ratio features).
* Encoded categorical features and scaled numeric features.
* Built an EDA notebook that exports visualizations (PNGs) into `eda_outputs/` for easy review.

## 🚀 How to run (local)

1. Ensure the dataset is at: `LLM_BACKEND_TEAM-B/backend/ml/data/transactions.csv`.

2. Run EDA (Jupyter notebook):

   ```bash
   cd LLM_BACKEND_TEAM-B/backend/ml
   jupyter notebook notebook/eda.ipynb
   ```

   EDA outputs (PNG/CSV) will be written to `LLM_BACKEND_TEAM-B/backend/ml/eda_outputs/`.

3. Run preprocessing script to generate cleaned CSV:

   ```bash
   cd LLM_BACKEND_TEAM-B/backend/ml
   python preprocessing.py
   ```

   Processed CSV: `LLM_BACKEND_TEAM-B/backend/ml/data/transactions_processed.csv`.

## 🧾 Output locations

* Raw data: `LLM_BACKEND_TEAM-B/backend/ml/data/transactions.csv`
* EDA visuals & summaries: `LLM_BACKEND_TEAM-B/backend/ml/eda_outputs/`
* Processed dataset: `LLM_BACKEND_TEAM-B/backend/ml/data/transactions_processed.csv`
* Trained models (future): `LLM_BACKEND_TEAM-B/backend/ml/models/`

## 🛠️ Next steps (high priority)

* Implement training pipeline in `training.py` and save model artifacts to `models/`.
* Add evaluation metrics and reports in `evaluation.py`.
* Create unit tests for preprocessing and EDA scripts.
* Add CI workflow to run tests and basic linting.

## 🧩 Conventions & Notes

* Column standardization: all column names are lowercased and non-alphanumeric characters replaced with underscores.
* Target column expected: `fraud_label` (script will raise an error if missing).
* Scripts are intentionally minimal and assume folders exist in the repo structure.

## 🗂️ Contact & Ownership

Maintained by: **FRONTEND TEAM B** — update `README.md` with correct owner/contact email.

