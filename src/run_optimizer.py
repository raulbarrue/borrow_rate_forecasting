"""
run_optimizer.py
-------------------------------------
Loads trained model, predicts next-day borrow rates using GBM,
runs optimizer, and saves optimal allocations.
"""

import os
import joblib
import pandas as pd
from src.model_train import prepare_features
from src.optimizer import optimize_inventory

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

synthetic_rates_path = os.path.join(DATA_DIR, 'synthetic_borrow_rates.csv')
market_data_path = os.path.join(DATA_DIR, 'market_data.csv')
model_path = os.path.join(MODEL_DIR, 'borrow_rate_model.pkl')

print("Loading model...")
model = joblib.load(model_path)

# Prepare latest features
rates = pd.read_csv(synthetic_rates_path, parse_dates=["Date"])
features_df = prepare_features(rates)
latest_date = features_df["Date"].max()
latest_features = features_df[features_df["Date"] == latest_date]

X_latest = latest_features[["RateLag1", "RateLag3", "RateLag5", "RollingVol", "Momentum"]]
tickers_latest = latest_features["Ticker"].values
X_latest_np = X_latest.to_numpy(dtype=float)

preds = model.predict(X_latest_np)
pred_df = pd.DataFrame({
    "Ticker": tickers_latest,
    "PredictedBorrowRate": preds
})

# Merge market data (optional)
market = pd.read_csv(market_data_path, parse_dates=["Date"])
latest_market = market.iloc[-1]
pred_df["LastClose"] = [latest_market[f"{t}_Close"] for t in tickers_latest]
pred_df["AvgDailyVolume"] = [market[f"{t}_Volume"].tail(20).mean() for t in tickers_latest]

alloc_df = optimize_inventory(pred_df, total_inventory=1_000_000)
alloc_path = os.path.join(DATA_DIR, "optimal_allocations.csv")
alloc_df.to_csv(alloc_path, index=False)

print(f"✅ Optimal allocations saved to {alloc_path}")
print(alloc_df)
