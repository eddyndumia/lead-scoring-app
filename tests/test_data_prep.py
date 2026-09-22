import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_prep import align_columns, get_feature_frame


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "source": ["Referral", "Cold outbound"],
        "industry": ["SaaS", "Retail"],
        "company_size": ["51-200", "1-10"],
        "job_title_seniority": ["VP/C-level", "IC"],
        "email_opens_30d": [8, 0],
        "site_visits_30d": [12, 1],
        "avg_pages_per_visit": [5.0, 1.0],
        "demo_requested": [1, 0],
        "budget_confirmed": [1, 0],
        "days_to_first_response": [0.5, 10.0],
    })


def test_get_feature_frame_one_hot_encodes(sample_df):
    X = get_feature_frame(sample_df)
    assert "source" not in X.columns
    assert any(c.startswith("source_") for c in X.columns)
    assert X.select_dtypes(include="object").empty


def test_align_columns_adds_missing_dummy_columns_as_zero():
    # a single-row frame only has dummy columns for the categories present
    # in that one row - align_columns must add every other trained column as 0
    single_row = pd.DataFrame({
        "source": ["Referral"], "industry": ["SaaS"], "company_size": ["51-200"],
        "job_title_seniority": ["VP/C-level"], "email_opens_30d": [8],
        "site_visits_30d": [12], "avg_pages_per_visit": [5.0],
        "demo_requested": [1], "budget_confirmed": [1], "days_to_first_response": [0.5],
    })
    X = get_feature_frame(single_row)
    reference_columns = list(X.columns) + ["source_Cold outbound", "industry_Retail"]
    aligned = align_columns(X, reference_columns)
    assert list(aligned.columns) == reference_columns
    assert aligned.loc[0, "source_Cold outbound"] == 0
    assert aligned.loc[0, "industry_Retail"] == 0


def test_align_columns_drops_unseen_columns():
    X = pd.DataFrame({"a": [1], "b": [2], "unseen_col": [3]})
    aligned = align_columns(X, reference_columns=["a", "b"])
    assert list(aligned.columns) == ["a", "b"]
    assert "unseen_col" not in aligned.columns


def test_get_feature_frame_preserves_row_count(sample_df):
    X = get_feature_frame(sample_df)
    assert len(X) == len(sample_df)
