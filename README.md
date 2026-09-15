# Customer Churn Prediction Pipeline

An end-to-end ML pipeline for predicting customer churn, built to demonstrate
production ML engineering practices: experiment tracking, CI/CD, and
monitoring — not just model training.

## Problem

Predict which customers are likely to cancel their subscription, using the
[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).


## What this project demonstrates

- End-to-end pipeline design, not just model fitting
- Experiment tracking and reproducibility
- CI/CD for ML (tests + automated retraining)
- Monitoring for data drift in production



## Model selection

- Compared performance of Logistic regression, Random forest and XGBoost with default hyperparameters using MLflow
- Logistic regression performed the best across roc_auc, precision, recall and f1 
- ROC AUC: [0.835, 0.807, 0.812]
- Precision: [0.671, 0.613, 0.61]
- Recall: [0.556, 0.471, 0.511]
- f1: [0.608, 0.533, 0.556]
- Experimented with different max depth for tree based models but even the best ones were only marginally better than logistic regression by some metrics and still mostly the same or worse when all metrics were taken into account
- This is likely due to the dataset being slightly too modest in size (7000 rows) for tree based models to significantly overtake. The features are mostly also more or less linear in correlation to the outcome (The longer the tenure, the more likely to stick with the company)
- Logistic regression was therefore used in the production pipeline although a full sweep of hyperparameters would be required in order to confirm it as the best option. This was not done in order to move on with the rest of the project