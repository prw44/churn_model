"""
Baseline model training for Telco Customer Churn prediction.

Usage:
    python src/train.py

TODO: implement the pipeline. See the spec you were given for the steps:
1. Load and clean data (watch TotalCharges)
2. Define features/target
3. Separate numeric vs categorical features
4. Build a ColumnTransformer + model Pipeline
5. Train/test split (stratified) and fit
6. Evaluate with classification_report + ROC AUC
"""


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


    
    
def load_data(filepath):
    data = pd.read_csv(filepath)
    
    X = data.drop(columns=["Churn", 'customerID'])
    Y = (data["Churn"] == "Yes").astype(int)

    return X, Y


def build_pipeline(num_cols, cat_cols):
    numerical_transformer = SimpleImputer(strategy='mean')
    
    categorical_transformer = Pipeline(steps = [('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown="ignore"))])
    
    preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)])
    
    model = LogisticRegression(max_iter=1000, random_state=0)
    
    return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])


def gen_pred_stats(pipeline, X_valid, Y_valid):
    # GENERATE PREDS AND PROBABILITIES OF CHURN
    preds = pipeline.predict(X_valid)
    probs = pipeline.predict_proba(X_valid)[:,1]

    print(classification_report(Y_valid, preds))
    print(f"ROC AUC: {roc_auc_score(Y_valid, probs):.4f}")
        

def main():

    X,Y = load_data("data/telco_data.csv")

    X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, train_size = 0.8, test_size = 0.2, stratify=Y, random_state=1)

    # DEAL WITH EMPTY STRINGS IN TOTALCHARGES COLUMN
    X_train['TotalCharges'] = pd.to_numeric(X_train['TotalCharges'], errors='coerce')
    X_valid['TotalCharges'] = pd.to_numeric(X_valid['TotalCharges'], errors='coerce')


    num_cols = X_train.select_dtypes(exclude = "object").columns.tolist()
    cat_cols = X_train.select_dtypes(include = "object").columns.tolist()


    pipeline = build_pipeline(num_cols, cat_cols)

    pipeline.fit(X_train, Y_train)


    gen_pred_stats(pipeline, X_valid, Y_valid)
    
    




if __name__ == "__main__":
    main()



