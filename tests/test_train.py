"""
Tests for src/train.py.
"""

from src.train import build_pipeline
from sklearn.linear_model import LogisticRegression
import pandas as pd


def test_pipeline_attributes():
    X = pd.DataFrame({'MonthlyCharges': [55, 65], 
                      'gender': ['male', 'female']})


    test_pipeline = build_pipeline(X ,LogisticRegression())
    assert hasattr(test_pipeline, 'fit')
    assert hasattr(test_pipeline, 'predict')


def test_pipeline_fits_on_toy_data():
    X = pd.DataFrame({"tenure": [1, 12, 24, 36],
                          "gender": ["Male", "Female", "Male", "Female"],})

    pipeline = build_pipeline(X, LogisticRegression())

    y = [0, 1, 0, 1]
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(y)



