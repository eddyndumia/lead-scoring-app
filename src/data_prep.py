"""Feature engineering and train/test split, shared by train.py and app.py."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "leads.csv"

CATEGORICAL_COLS = ["source", "industry", "company_size", "job_title_seniority"]
NUMERIC_COLS = [
    "email_opens_30d", "site_visits_30d", "avg_pages_per_visit",
    "demo_requested", "budget_confirmed", "days_to_first_response",
]
TARGET = "converted"
ALL_INPUT_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def get_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df[ALL_INPUT_COLS], columns=CATEGORICAL_COLS, drop_first=False)


def align_columns(X: pd.DataFrame, reference_columns: list) -> pd.DataFrame:
    """
    Makes a single new lead's one-hot-encoded row match the exact column set
    the model was trained on (adds any missing dummy columns as 0, drops any
    the model has never seen). Needed because get_dummies on a single row
    only produces columns for the categories present in that one row.
    """
    return X.reindex(columns=reference_columns, fill_value=0)


def split(df: pd.DataFrame, test_size=0.2, random_state=42):
    X = get_feature_frame(df)
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
