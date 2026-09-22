"""
Generates synthetic-but-realistic inbound lead data with a genuine,
noisy conversion signal built from real B2B lead-scoring factors:
firmographic fit (company size, industry) and behavioral engagement
(email opens, site visits, demo requested, budget confirmed, response
speed). Synthetic by design - reproducible numbers, no real CRM data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(21)
N = 6000
OUT = Path(__file__).resolve().parent.parent / "data" / "leads.csv"

SOURCES = ["Website form", "Content download", "Webinar", "Cold outbound", "Referral", "Paid ad"]
SOURCE_WEIGHTS = [0.28, 0.18, 0.12, 0.22, 0.10, 0.10]
INDUSTRIES = ["SaaS", "Retail", "Manufacturing", "Healthcare", "Finance", "Education"]
COMPANY_SIZES = ["1-10", "11-50", "51-200", "201-1000", "1000+"]
SIZE_WEIGHTS = [0.30, 0.30, 0.22, 0.13, 0.05]
# rough ICP (ideal customer profile) fit by size - mid-market is the sweet spot for this product
SIZE_FIT = {"1-10": 0.3, "11-50": 0.75, "51-200": 1.0, "201-1000": 0.85, "1000+": 0.5}
SOURCE_QUALITY = {"Website form": 0.9, "Content download": 0.6, "Webinar": 0.8,
                   "Cold outbound": 0.35, "Referral": 1.1, "Paid ad": 0.5}


def generate() -> pd.DataFrame:
    company_size = RNG.choice(COMPANY_SIZES, size=N, p=SIZE_WEIGHTS)
    industry = RNG.choice(INDUSTRIES, size=N)
    source = RNG.choice(SOURCES, size=N, p=SOURCE_WEIGHTS)

    email_opens_30d = RNG.poisson(3, size=N)
    site_visits_30d = RNG.poisson(4, size=N)
    pages_per_visit = RNG.gamma(2, 1.3, size=N).round(1)
    demo_requested = RNG.choice([0, 1], size=N, p=[0.78, 0.22])
    budget_confirmed = RNG.choice([0, 1], size=N, p=[0.85, 0.15])
    days_to_first_response = RNG.exponential(3, size=N).clip(0, 30).round(1)
    job_title_seniority = RNG.choice(["IC", "Manager", "Director", "VP/C-level"], size=N,
                                      p=[0.40, 0.32, 0.20, 0.08])

    seniority_weight = {"IC": 0.0, "Manager": 0.3, "Director": 0.6, "VP/C-level": 0.9}

    score = np.zeros(N)
    score += np.array([SIZE_FIT[s] for s in company_size]) * 1.4
    score += np.array([SOURCE_QUALITY[s] for s in source]) * 1.1
    score += np.minimum(email_opens_30d, 8) * 0.14
    score += np.minimum(site_visits_30d, 10) * 0.11
    score += demo_requested * 1.5
    score += budget_confirmed * 1.3
    score += np.array([seniority_weight[t] for t in job_title_seniority]) * 1.0
    score -= np.minimum(days_to_first_response, 14) * 0.09  # slower first response = colder lead
    score += RNG.normal(0, 1.1, size=N)  # noise

    prob = 1 / (1 + np.exp(-(score - 5.2)))  # centered for a realistic ~18% conversion rate
    converted = (RNG.uniform(0, 1, size=N) < prob).astype(int)

    df = pd.DataFrame({
        "lead_id": [f"LEAD-{i:05d}" for i in range(1, N + 1)],
        "source": source,
        "industry": industry,
        "company_size": company_size,
        "job_title_seniority": job_title_seniority,
        "email_opens_30d": email_opens_30d,
        "site_visits_30d": site_visits_30d,
        "avg_pages_per_visit": pages_per_visit,
        "demo_requested": demo_requested,
        "budget_confirmed": budget_confirmed,
        "days_to_first_response": days_to_first_response,
        "converted": converted,
    })
    return df


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} leads -> {OUT}")
    print(f"Conversion rate: {df['converted'].mean():.1%}")


if __name__ == "__main__":
    main()
