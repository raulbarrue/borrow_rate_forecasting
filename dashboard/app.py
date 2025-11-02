"""
dashboard/app.py
-------------------------------------
Streamlit dashboard for Prime Finance prototype:
- Displays predicted borrow rates
- Shows optimized allocations
- Includes "Trader Assist Mode" to auto-refresh predictions & allocations
"""

import os
import time
import joblib
import pandas as pd
import streamlit as st
from src.model_train import prepare_features
from src.optimizer import optimize_inventory

# --- CONFIGURATION ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

synthetic_rates_path = os.path.join(DATA_DIR, 'synthetic_borrow_rates.csv')
market_data_path = os.path.join(DATA_DIR, 'market_data.csv')
model_path = os.path.join(MODEL_DIR, 'borrow_rate_model.pkl')


# --- CORE LOGIC ---
@st.cache_data
def load_data():
    rates = pd.read_csv(synthetic_rates_path, parse_dates=["Date"])
    market = pd.read_csv(market_data_path, parse_dates=["Date"])
    return rates, market


def get_latest_predictions(model, rates, market):
    features_df = prepare_features(rates)
    latest_date = features_df["Date"].max()
    latest_features = features_df[features_df["Date"] == latest_date]

    X_latest = latest_features[["RateLag1", "RateLag3", "RateLag5", "RollingVol", "Momentum"]]
    tickers_latest = latest_features["Ticker"].values
    preds = model.predict(X_latest.to_numpy(dtype=float))

    pred_df = pd.DataFrame({
        "Ticker": tickers_latest,
        "PredictedBorrowRate": preds
    })

    latest_market = market.iloc[-1]
    pred_df["LastClose"] = [latest_market[f"{t}_Close"] for t in tickers_latest]
    pred_df["AvgDailyVolume"] = [market[f"{t}_Volume"].tail(20).mean() for t in tickers_latest]

    return pred_df


def run_optimization(model, rates, market):
    pred_df = get_latest_predictions(model, rates, market)
    alloc_df = optimize_inventory(pred_df, total_inventory=1_000_000)
    return pred_df, alloc_df


# --- STREAMLIT UI ---
st.set_page_config(page_title="Prime Finance Optimizer", layout="wide")
st.title("📊 Prime Finance Borrow Rate Forecast & Inventory Optimizer")

model = joblib.load(model_path)
rates, market = load_data()

# Sidebar controls
st.sidebar.header("Configuration")
refresh_interval = st.sidebar.slider("Trader Assist Refresh (seconds)", 5, 60, 15)
assist_mode = st.sidebar.toggle("Enable Trader Assist Mode", value=False)

# Run initial optimization
pred_df, alloc_df = run_optimization(model, rates, market)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Predicted Borrow Rates")
    st.dataframe(pred_df.sort_values("PredictedBorrowRate", ascending=False).reset_index(drop=True))

with col2:
    st.subheader("Optimized Inventory Allocation")
    st.dataframe(alloc_df)

# Trader Assist Mode
if assist_mode:
    st.markdown("---")
    st.info("Trader Assist Mode active — refreshing every few seconds...")
    placeholder_rates = st.empty()
    placeholder_alloc = st.empty()

    while True:
        time.sleep(refresh_interval)
        pred_df, alloc_df = run_optimization(model, rates, market)

        placeholder_rates.dataframe(pred_df.sort_values("PredictedBorrowRate", ascending=False).reset_index(drop=True))
        placeholder_alloc.dataframe(alloc_df)

        st.toast(f"Refreshed allocations at {time.strftime('%H:%M:%S')}", icon="🔄")
