import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.model_selection import train_test_split

def cat_drift(df_ref, df_new):
    cat_stats = {}

    for col in df_ref.columns:
        ref_counts = df_ref[col].value_counts()
        new_counts = df_new[col].value_counts()

        contingency_table = pd.DataFrame({
        "reference": ref_counts,
        "new": new_counts}).fillna(0)

        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        cat_stats[col] = {"chi2": chi2, "p_value": p_value}

    return cat_stats


def num_drift(df_ref, df_new):
    num_stats = {}

    for col in df_ref.columns:
        stat, p_value = ks_2samp(df_ref[col], df_new[col])
        num_stats[col] = {"statistic": stat, "p_value": p_value}

    return num_stats



def drift_detection(df_ref, df_new):

    num_cols = df_ref.select_dtypes(exclude = "object").columns.tolist()
    cat_cols = df_ref.select_dtypes(include = "object").columns.tolist()

    num_stats = num_drift(df_ref[num_cols], df_new[num_cols])
    cat_stats = cat_drift(df_ref[cat_cols], df_new[cat_cols])

    
    return {"categorical": cat_stats, "numerical": num_stats}


    


def drift_summary(drift_stats, threshold):

    drifted_columns = []
    for group in drift_stats.values():
        for col, stats in group.items():
            if stats['p_value'] <= threshold:
                drifted_columns.append(col)
    return drifted_columns



def main():

    """
    Demonstrates the drift detection logic using a random train/test
    split of the existing dataset as a stand-in for "new" data, since no
    real time-separated data source is available. This validates that the
    detection logic works correctly — it is not a live production drift
    check. See README for more detail.
    """
    
    df = pd.read_csv('data/telco_data.csv')

    X = df.drop(columns=["Churn", 'customerID'])
    Y = (df["Churn"] == "Yes").astype(int)
    
    X_ref, X_new, Y_ref, Y_new = train_test_split(X,Y, train_size = 0.8, test_size = 0.2, stratify=Y, random_state=1)

    stats = drift_detection(X_ref, X_new)
    drifted_columns = drift_summary(stats, 0.05)

    if drifted_columns:
        print(f"The columns which drifted are {drifted_columns}")

    else: 
        print("No columns drifted")


    return stats, drifted_columns



if __name__ == "__main__":
    main() 

