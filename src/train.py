"""
Trains a lead-scoring classifier and, critically, picks its decision
threshold based on real sales team capacity rather than defaulting to 0.5.

If a sales team can realistically follow up on 25% of the leads coming in
each week, the right question isn't "what's this model's accuracy at 0.5?"
- it's "what score cutoff gives us exactly that many leads, and what
fraction of actual converters does that cutoff catch?" That's what
reports/capacity_thresholds.csv answers.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.data_prep import get_feature_frame, load_raw, split

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
CAPACITY_LEVELS = [0.10, 0.15, 0.20, 0.25, 0.35, 0.50]  # fraction of leads sales can contact


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    X_train, X_test, y_train, y_test = split(df)

    model = GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    print(f"ROC-AUC: {auc:.4f}   Average precision: {ap:.4f}")

    # --- capacity-based thresholds: rank every test lead by score, and for each
    # "sales can contact the top X%" level, report the score cutoff, precision,
    # and recall at that cutoff ---
    ranked = pd.DataFrame({"y_true": y_test.values, "score": proba}).sort_values("score", ascending=False)
    n = len(ranked)
    total_converters = ranked["y_true"].sum()

    rows = []
    for capacity in CAPACITY_LEVELS:
        k = max(1, int(round(n * capacity)))
        top_k = ranked.iloc[:k]
        precision = top_k["y_true"].mean()
        recall = top_k["y_true"].sum() / total_converters
        rows.append({
            "capacity_pct": int(capacity * 100),
            "leads_contacted": k,
            "score_cutoff": round(top_k["score"].min(), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        })
    capacity_df = pd.DataFrame(rows)
    capacity_df.to_csv(REPORTS_DIR / "capacity_thresholds.csv", index=False)
    print("\nCapacity-based thresholds:")
    print(capacity_df.to_string(index=False))

    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    pd.DataFrame({
        "threshold": list(thresholds) + [1.0],
        "precision": precisions,
        "recall": recalls,
    }).to_csv(REPORTS_DIR / "precision_recall_curve.csv", index=False)

    joblib.dump(model, REPORTS_DIR / "lead_scoring_model.joblib")
    with open(REPORTS_DIR / "model_meta.json", "w") as f:
        json.dump({
            "roc_auc": round(auc, 4),
            "average_precision": round(ap, 4),
            "feature_columns": list(X_train.columns),
            "base_conversion_rate": round(float(df["converted"].mean()), 4),
        }, f, indent=2)

    print(f"\nSaved model -> {REPORTS_DIR / 'lead_scoring_model.joblib'}")


if __name__ == "__main__":
    main()
