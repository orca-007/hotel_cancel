# Hotel Booking Cancellation Risk & Overbooking Strategy

BAMD project (IIM Calcutta) — a booking-time cancellation-risk model for a hotel group operating one City Hotel and one Resort Hotel, turned into a concrete overbooking and retention-outreach policy.

## What's in this folder

| File / folder | What it is |
|---|---|
| `Project1_Hotel_Cancellation_Risk.ipynb` | The full analysis notebook — cleaning, leakage audit, EDA, four models compared on a time-based split, calibration check, risk segmentation, and an overbooking simulator. Already executed; open it and read top to bottom, or re-run it yourself. |
| `app.py` | The Streamlit dashboard — the same analysis, packaged for a Revenue Manager to use interactively. |
| `pipeline.py` | The shared analytics engine. Both the notebook and the dashboard import this module, so neither can silently disagree with the other. |
| `data/hotel_bookings.csv` | The raw dataset (Kaggle: Hotel Booking Demand, Antonio/Almeida/Nunes 2019). |
| `models/` | Pre-trained models and processed train/test data, saved so the dashboard loads in under a second instead of retraining on every visit. Delete this folder and the app will retrain from scratch automatically. |
| `requirements.txt` | Python dependencies. |
| `.streamlit/config.toml` | Dashboard color theme. |

## Running the dashboard locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL it prints (usually `http://localhost:8501`).

## Deploying to Streamlit Community Cloud

1. Push this entire folder to a GitHub repository (keep the folder structure as-is — `app.py` must sit at the repo root, with `pipeline.py`, `data/`, and `models/` alongside it).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and click **New app**.
3. Point it at your repository and set the main file path to `app.py`.
4. Deploy. First load may take a few seconds longer if `models/` wasn't included in the repo (it will retrain automatically); after that, every load is fast.

## Re-running the notebook

```bash
pip install -r requirements.txt jupyter
jupyter notebook Project1_Hotel_Cancellation_Risk.ipynb
```

Run all cells top to bottom. The final cell overwrites `models/` with freshly trained artifacts — run it again any time the underlying data or pipeline logic changes, and the dashboard will pick up the new numbers automatically on its next load.
