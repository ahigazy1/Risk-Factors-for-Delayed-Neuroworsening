import pandas as pd
import numpy as np
import re
from tbi_dnw.config import PUBLICATION_LABELS

def load_and_clean_data(config):
    """
    Load and preprocess the TBI dataset.
    """
    print(f"Loading data from: {config['DATA_PATH']}")

    # Load data
    try:
        df = pd.read_csv(
            config['DATA_PATH'],
            na_values=config['NA_VALUES'],
            skipinitialspace=True
        )
    except FileNotFoundError:
        print("ERROR: File not found. Please check the path.")
        return None, None

    # Clean column names
    df.columns = df.columns.str.strip().str.replace('[^A-Za-z0-9_]+', '', regex=True)
    target_clean = re.sub(r'[^A-Za-z0-9_]+', '', config['TARGET_COL'].strip())

    # Drop administrative columns
    drop_candidates = ['0824_Record_ID', 'Trauma_Cohort_Year', 'Early_NW_YN']
    found_drops = [c for c in drop_candidates if c in df.columns]
    if found_drops:
        df = df.drop(columns=found_drops)

    # Type conversions
    # Legal Sex:  Male=1, Female=0
    if 'Legal_Sex' in df.columns:
        df['Legal_Sex'] = df['Legal_Sex'].astype(str).str.strip()
        df['Legal_Sex'] = df['Legal_Sex'].map({'Male': 1, 'Female': 0})

    # Force numeric for GCS and BP columns
    num_cols = [
        'Admission_Diastolic', 'Admission_Systolic',
        'EMS_GCS', 'Admit_GCS', 'NSGY_GCS',
        'EMS_MOTOR_GCS', 'Admit_Motor_GCS', 'NSGY_Motor_GCS'
    ]

    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # One-Hot Encoding
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, drop_first=True)

    # Handle missing values
    imputable_cols = [c for c in num_cols if c in df.columns]
    non_imputable_cols = [c for c in df.columns if c not in imputable_cols and c != target_clean]

    initial_rows = len(df)
    df = df.dropna(subset=non_imputable_cols)
    dropped_count = initial_rows - len(df)

    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows due to missing data in non-imputable columns.")

    y = df[target_clean].values
    X = df.drop(columns=[target_clean])
    X = X.rename(columns=PUBLICATION_LABELS)

    print(f"Data loaded successfully. Features shape: {X.shape}, Target prevalence: {y.mean():.2%}")

    return X, y
