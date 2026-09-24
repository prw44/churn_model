# Customer Churn Prediction Pipeline

An end-to-end ML pipeline for predicting customer churn, built to demonstrate
production ML engineering practices: experiment tracking, CI/CD, and
monitoring — not just model training.


## Live demo

Try the churn prediction dashboard here: https://churnmodel-bjqifr5ovnttvsyrg7zfhk.streamlit.app/





## Problem

Predict which customers are likely to cancel their subscription, using the
[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).


## What this project demonstrates

- End-to-end pipeline design, not just model fitting
- Experiment tracking and reproducibility
- CI/CD for ML (tests + automated retraining)
- Monitoring for data drift in production


## Roadmap

- [x] Baseline model + comparison (logistic regression, random forest, XGBoost)
- [x] Experiment tracking with MLflow
- [x] Automated retraining via GitHub Actions
- [x] Data drift detection (tested, not wired to a live data source)
- [x] Monitoring dashboard (Streamlit)
- [x] Deployment



## Model selection

- Compared performance of Logistic regression, Random forest and XGBoost with default hyperparameters using MLflow
- Logistic regression performed the best across roc_auc, precision, recall and f1 
- Experimented with different max depth for tree based models but even the best ones were only marginally better than logistic regression by some metrics and still mostly the same or worse when all metrics were taken into account
- This is likely due to the dataset being slightly too modest in size (7000 rows) for tree based models to significantly overtake. The features are mostly also more or less linear in correlation to the outcome (The longer the tenure, the more likely to stick with the company)
- Logistic regression was therefore used in the production pipeline although a full sweep of hyperparameters would be required in order to confirm it as the best option. This was not done in order to move on with the rest of the project



## CI implementation

- In github workflows, I added a retrain job triggered by changes in the data folder.


## Drift detection

- I conducted research into data drift and the best metrics and statistical tests to use for detecting it
- I decided to use the two-sample Kolmogorov-Smirnov test for numerical columns and a Chi-square contingency test as the statistical tests and just use p value as the main metric although this could easily be altered or expanded in the future
- Since there is no continous inflow of data, the drift detection logic doesn't actually do anything but is fully tested and so ready to be implemented if there was live data
- The main function shows the detection in action on a train test style split of data although it naturally doesn't detect drift within this data


## Streamlit monitoring dashboard

- Built an interactive dashboard in which a user inputs customer data via a sidebar and it outputs both a churn prediction and probability
- The CI retrain job now saves the retrained model to `model.pkl` and automatically commits it back to the repo, so the dashboard always serves the most recently trained model without manual steps
- Deployed the dashboard using streamlit





