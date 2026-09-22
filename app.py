"""
Streamlit app: score a single lead by hand, or upload a CSV of leads and
get them ranked with a recommended contact/no-contact call based on your
team's actual weekly capacity.

Run: streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data_prep import ALL_INPUT_COLS, CATEGORICAL_COLS, align_columns, get_feature_frame

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
MODEL_PATH = REPORTS_DIR / "lead_scoring_model.joblib"
META_PATH = REPORTS_DIR / "model_meta.json"
CAPACITY_PATH = REPORTS_DIR / "capacity_thresholds.csv"

st.set_page_config(page_title="Lead Scoring", page_icon="📈", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


@st.cache_data
def load_capacity_table():
    return pd.read_csv(CAPACITY_PATH)


def score_leads(df: pd.DataFrame, model, feature_columns) -> pd.Series:
    X = get_feature_frame(df)
    X = align_columns(X, feature_columns)
    return model.predict_proba(X)[:, 1]


def main():
    model, meta = load_model()
    capacity_df = load_capacity_table()

    st.title("Lead Scoring")
    st.caption(
        f"Model: gradient boosting classifier, ROC-AUC {meta['roc_auc']:.2f} on held-out test data. "
        f"Base conversion rate in training data: {meta['base_conversion_rate']:.1%}."
    )

    tab_single, tab_batch, tab_capacity = st.tabs(["Score one lead", "Score a CSV", "Capacity guide"])

    with tab_single:
        st.subheader("Enter lead details")
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Source", ["Website form", "Content download", "Webinar",
                                               "Cold outbound", "Referral", "Paid ad"])
            industry = st.selectbox("Industry", ["SaaS", "Retail", "Manufacturing",
                                                   "Healthcare", "Finance", "Education"])
            company_size = st.selectbox("Company size", ["1-10", "11-50", "51-200", "201-1000", "1000+"])
            seniority = st.selectbox("Contact seniority", ["IC", "Manager", "Director", "VP/C-level"])
        with col2:
            email_opens = st.number_input("Email opens (last 30d)", 0, 50, 3)
            site_visits = st.number_input("Site visits (last 30d)", 0, 50, 4)
            pages_per_visit = st.number_input("Avg pages per visit", 0.0, 20.0, 2.5, step=0.1)
            days_to_response = st.number_input("Days to first response", 0.0, 30.0, 2.0, step=0.5)
        demo_requested = st.checkbox("Demo requested")
        budget_confirmed = st.checkbox("Budget confirmed")

        if st.button("Score this lead", type="primary"):
            row = pd.DataFrame([{
                "source": source, "industry": industry, "company_size": company_size,
                "job_title_seniority": seniority, "email_opens_30d": email_opens,
                "site_visits_30d": site_visits, "avg_pages_per_visit": pages_per_visit,
                "demo_requested": int(demo_requested), "budget_confirmed": int(budget_confirmed),
                "days_to_first_response": days_to_response,
            }])
            score = score_leads(row, model, meta["feature_columns"])[0]
            st.metric("Conversion likelihood", f"{score:.1%}")
            rank_pct = (capacity_df["score_cutoff"] <= score).sum()
            if score >= capacity_df.iloc[3]["score_cutoff"]:  # 25% capacity row
                st.success("Above the top-25%-capacity cutoff — prioritize this lead.")
            elif score >= capacity_df.iloc[-1]["score_cutoff"]:
                st.info("Mid-tier — contact if capacity allows beyond the top 25%.")
            else:
                st.warning("Below the top-50%-capacity cutoff — low priority given typical sales bandwidth.")

    with tab_batch:
        st.subheader("Upload a CSV of leads")
        st.caption(f"Expected columns: {', '.join(ALL_INPUT_COLS)}")
        uploaded = st.file_uploader("CSV file", type="csv")
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            missing = set(ALL_INPUT_COLS) - set(batch_df.columns)
            if missing:
                st.error(f"Missing required columns: {sorted(missing)}")
            else:
                batch_df["conversion_score"] = score_leads(batch_df, model, meta["feature_columns"])
                batch_df = batch_df.sort_values("conversion_score", ascending=False).reset_index(drop=True)
                batch_df["rank"] = batch_df.index + 1
                st.dataframe(batch_df, use_container_width=True)
                st.download_button(
                    "Download scored + ranked CSV",
                    batch_df.to_csv(index=False).encode("utf-8"),
                    "scored_leads.csv", "text/csv",
                )

    with tab_capacity:
        st.subheader("What conversion rate should you expect at your team's real capacity?")
        st.caption(
            "If your sales team can realistically contact X% of incoming leads each week, "
            "this is the precision/recall you'd get by always contacting the model's top-scored X%, "
            "measured on held-out test data."
        )
        st.dataframe(capacity_df, use_container_width=True, hide_index=True)
        st.caption(
            f"At the 25% capacity row: contacting the top-scored quarter of leads catches "
            f"{capacity_df.iloc[3]['recall']:.0%} of actual converters at "
            f"{capacity_df.iloc[3]['precision']:.0%} precision — "
            f"a {capacity_df.iloc[3]['precision'] / meta['base_conversion_rate']:.1f}x lift "
            f"over contacting a random 25% of leads."
        )


if __name__ == "__main__":
    main()
