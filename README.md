# Lead Scoring App

A lead-scoring model in a Streamlit app. Score one lead, upload a CSV to rank a batch, or see what your team gets if it can only call some of its leads.

The data is synthetic (`src/generate_data.py`), built from company size, industry, lead source and engagement, with an 18.6% conversion rate.

ROC-AUC is 0.70, but that means nothing to a sales manager. This does:

| Team can call | Precision | Converters caught | vs random |
|---|---|---|---|
| top 10% | 46.7% | 25.1% | 2.5x |
| **top 25%** | **35.7%** | **48.0%** | **1.9x** |
| top 50% | 27.2% | 73.1% | 1.5x |

![Capacity tradeoff](reports/figures/capacity_tradeoff.png)

A small bug worth knowing about: one-hot encoding a single new lead only makes columns for that lead's own categories, so the model gets the wrong shape. `align_columns` in `src/data_prep.py` fixes it.

```bash
pip install -r requirements.txt
python src/generate_data.py
python -m src.train
streamlit run app.py
```

To put it online, connect the repo at [share.streamlit.io](https://share.streamlit.io) and point it at `app.py`.
