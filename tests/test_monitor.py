"""
Tests for src/monitor.py
"""

import pandas as pd
from src.monitor import drift_detection, drift_summary




def test_drift_detected_on_shifted_numerical_data():

    df = pd.DataFrame({"MonthlyCharges": [50, 42, 60, 79, 58] * 20})
    df_new = df.copy()
    df_new["MonthlyCharges"] += 30

    stats = drift_detection(df, df_new)
    assert drift_summary(stats, 0.05) == ["MonthlyCharges"]



def test_drift_detected_on_shifted_categorical_data():

    df = pd.DataFrame({'gender': ['male', 'male', 'female', 'male'] * 20})
    df_new = pd.DataFrame({'gender': ['female', 'male', 'female', 'female'] * 20})

    stats = drift_detection(df, df_new)
    assert drift_summary(stats, 0.05) == ['gender']



def test_no_drift_detected_on_undrifted_data():
    df = pd.DataFrame({"MonthlyCharges": [50, 42, 60, 79, 58] * 20})
    df_new = df.copy()
    
    stats = drift_detection(df, df_new)
    assert drift_summary(stats, 0.05) == []



        