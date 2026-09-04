# Customer Churn Prediction Pipeline

An end-to-end ML pipeline for predicting customer churn, built to demonstrate
production ML engineering practices: experiment tracking, CI/CD, and
monitoring — not just model training.

## Problem

Predict which customers are likely to cancel their subscription, using the
[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Project structure

```
├── data/           # raw data (gitignored — see setup below)
├── notebooks/      # exploratory data analysis
├── src/            # pipeline code
├── tests/          # unit tests
└── .github/workflows/  # CI pipeline
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Download the dataset from Kaggle and place it at `data/telco_churn.csv`.

## Usage

Train the baseline model:

```bash
python src/train.py
```

Run tests:

```bash
pytest tests/ -v
```

## Roadmap

- [ ] Baseline model (logistic regression)
- [ ] Experiment tracking with MLflow
- [ ] Automated retraining via GitHub Actions
- [ ] Data drift detection
- [ ] Monitoring dashboard (Streamlit)
- [ ] Deployment

## What this project demonstrates

- End-to-end pipeline design, not just model fitting
- Experiment tracking and reproducibility
- CI/CD for ML (tests + automated retraining)
- Monitoring for data drift in production
