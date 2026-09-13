"""

Tests for src/train.py.

TODO: write tests once your pipeline exists, e.g.:
- does your pipeline-building function return something with .fit/.predict?
- does it fit without error on a small toy DataFrame?
"""

from src.train import load_data, build_pipeline, gen_pred_stats
import pandas as pd


def test_pipeline_attributes():
    test_pipeline = build_pipeline(['MonthlyCharges'], ['InternetService'])
    assert hasattr(test_pipeline, 'fit')
    assert hasattr(test_pipeline, 'predict')


def test_pipeline_fits_on_toy_data():
    pipeline = build_pipeline(["tenure"], ["gender"])
    X = pd.DataFrame({"tenure": [1, 12, 24, 36],
                      "gender": ["Male", "Female", "Male", "Female"],})

    y = [0, 1, 0, 1]
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(y)



def test_file_read():
    X, Y = load_data("data/telco_data.csv")
    assert Y.isin([0,1]).all()