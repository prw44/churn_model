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

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import joblib

import mlflow
import mlflow.sklearn
    


def load_split_data(filepath):
    data = pd.read_csv(filepath)
    
    X = data.drop(columns=["Churn", 'customerID'])
    Y = (data["Churn"] == "Yes").astype(int)

    X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y, train_size = 0.8, test_size = 0.2, stratify=Y, random_state=1)
    
    X_train = X_train.copy()
    X_valid = X_valid.copy()

    # DEAL WITH EMPTY STRINGS IN TOTALCHARGES COLUMN
    X_train['TotalCharges'] = pd.to_numeric(X_train['TotalCharges'], errors='coerce')
    X_valid['TotalCharges'] = pd.to_numeric(X_valid['TotalCharges'], errors='coerce')
    
    return X_train, X_valid, Y_train, Y_valid


def build_preprocessor(df):

    num_cols = df.select_dtypes(exclude = "object").columns.tolist()
    cat_cols = df.select_dtypes(include = "object").columns.tolist()

    numerical_transformer = Pipeline(steps = [('impute', SimpleImputer(strategy='mean', add_indicator=True)), 
                                              ('scale', StandardScaler())]) 
                                                    
    categorical_transformer = Pipeline(steps = [('impute', SimpleImputer(strategy='most_frequent')), 
                                                ('onehot', OneHotEncoder(handle_unknown="ignore"))])
        
    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, num_cols),
                                                   ('cat', categorical_transformer, cat_cols)])

    return preprocessor




def build_pipeline(X_train, classifier):
  
    preprocessor = build_preprocessor(X_train)

    model = classifier

    return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])



def gen_preds_probs(pipeline, X_valid):
    # GENERATE PREDS AND PROBABILITIES OF CHURN
    preds = pipeline.predict(X_valid)
    probs = pipeline.predict_proba(X_valid)[:,1]


    return preds, probs


def main():

    X_train, X_valid, Y_train, Y_valid = load_split_data("data/telco_data.csv")

    trial_models = {'logistic_regression': LogisticRegression(random_state=0)}


    for name, model in trial_models.items():

        with mlflow.start_run(run_name = name):


            pipeline = build_pipeline(X_train, classifier=model)

            params = pipeline.named_steps['model'].get_params()

            pipeline.fit(X_train, Y_train)


            preds, probs = gen_preds_probs(pipeline, X_valid)
    

            mlflow.log_params(params)
            mlflow.log_metrics({'roc_auc': roc_auc_score(Y_valid, probs), 
                                'f1': f1_score(Y_valid, preds),
                                'precision': precision_score(Y_valid, preds),
                                'recall': recall_score(Y_valid, preds)})
            mlflow.sklearn.log_model(pipeline, "model", serialization_format='pickle')


    joblib.dump(pipeline, "model.pkl")



if __name__ == "__main__":
    main()



