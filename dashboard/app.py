"""
dashboard/app.py
-------------------------------------
Streamlit dashboard for Prime Finance prototype:
- Displays predicted borrow rates
- Shows optimized allocations
- Includes "Trader Assist Mode" to auto-refresh predictions & allocations
- Adds offline Trader Assist numerical tools: risk calculators, charts, trade planning, simulations
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
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

# --- Existing Borrow Rate & Allocation Display ---
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

# --- New Trader Assist Numerical Features in Tabs ---
st.markdown("---")
st.header("Trader Assist Tools")
tabs = st.tabs([
    "Portfolio & Risk",
 #   "Charts & Visualization",
 #   "Trade Planning",
    "Scenario Simulations"
 #   "Tips & Insights"
])

# ----------------------
# 2️⃣ Portfolio & Risk Management
# ----------------------
with tabs[0]:
    st.subheader("Position Size & Risk/Reward Calculator")
    capital = st.number_input("Capital ($)", value=100000)
    risk_percent = st.number_input("Risk per Trade (%)", value=1.0)
    entry_price = st.number_input("Entry Price ($)", value=100.0)
    stop_price = st.number_input("Stop-Loss Price ($)", value=95.0)
    target_price = st.number_input("Target Price ($)", value=110.0)

    position_size = (capital * (risk_percent / 100)) / max(entry_price - stop_price, 0.01)
    st.write(f"Recommended Position Size: **{position_size:.2f} units**")

    rrr = (target_price - entry_price) / max(entry_price - stop_price, 0.01)
    st.write(f"Risk/Reward Ratio: **{rrr:.2f}**")

    # st.subheader("Sample Portfolio Diversification")
    # portfolio_df = pd.DataFrame({
    #     "Asset": ["Stock A", "Stock B", "ETF C", "Bond D"],
    #     "Allocation": [30, 25, 25, 20]
    # })
    # fig, ax = plt.subplots()
    # ax.pie(portfolio_df['Allocation'], labels=portfolio_df['Asset'], autopct='%1.0f%%')
    # st.pyplot(fig)

    st.subheader("Trade Fees Calculator")
    shares = st.number_input("Number of Shares", value=int(position_size))
    commission = st.number_input("Commission per Trade ($)", value=10.0)
    slippage = st.number_input("Slippage (%)", value=0.1)
    total_cost = shares*entry_price + commission + (shares*entry_price*slippage/100)
    st.write(f"Total Cost of Trade: **${total_cost:.2f}**")

# # ----------------------
# # 3️⃣ Charts / Visualization
# # ----------------------
# with tabs[1]:
#     st.subheader("Sample Price Chart")
#     dates = pd.date_range(start="2025-01-01", periods=100)
#     prices = np.cumsum(np.random.randn(100)) + 100
#     price_df = pd.DataFrame({"Date": dates, "Price": prices})
#     fig, ax = plt.subplots()
#     ax.plot(price_df['Date'], price_df['Price'])
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Price")
#     st.pyplot(fig)

#     st.subheader("Comparison Chart")
#     prices2 = np.cumsum(np.random.randn(100)) + 80
#     fig, ax = plt.subplots()
#     ax.plot(dates, prices, label="Asset A")
#     ax.plot(dates, prices2, label="Asset B")
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Price")
#     ax.legend()
#     st.pyplot(fig)

# # ----------------------
# # 4️⃣ Trade Planning / Journaling
# # ----------------------
# with tabs[2]:
#     st.subheader("Trade Log")
#     trade_log = pd.DataFrame(columns=["Asset", "Entry", "Stop", "Target", "Position Size"])
#     asset_name = st.text_input("Asset Name", value="Stock A")
#     if st.button("Add Trade"):
#         new_trade = pd.DataFrame([{
#             "Asset": asset_name,
#             "Entry": entry_price,
#             "Stop": stop_price,
#             "Target": target_price,
#             "Position Size": position_size
#         }])
#         trade_log = pd.concat([trade_log, new_trade], ignore_index=True)
#         st.success("Trade added!")

#     st.subheader("Pre-Trade Checklist")
#     st.checkbox("Check Trend Direction")
#     st.checkbox("Confirm Stop-Loss Level")
#     st.checkbox("Verify Capital Allocation")
#     st.checkbox("Review Risk/Reward")

#     st.subheader("Strategy Tester (SMA Crossover)")
#     short_window = 5
#     long_window = 20
#     price_df['SMA_short'] = price_df['Price'].rolling(short_window).mean()
#     price_df['SMA_long'] = price_df['Price'].rolling(long_window).mean()
#     price_df['Signal'] = np.where(price_df['SMA_short'] > price_df['SMA_long'], 1, 0)
#     price_df['Daily Return'] = price_df['Price'].pct_change()
#     price_df['Strategy Return'] = price_df['Signal'].shift(1) * price_df['Daily Return']
#     cumulative_return = (1 + price_df['Strategy Return'].fillna(0)).cumprod()
#     fig, ax = plt.subplots()
#     ax.plot(price_df['Date'], cumulative_return, label="Strategy Cumulative Return")
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Cumulative Return")
#     ax.legend()
#     st.pyplot(fig)

# ----------------------
# 5️⃣ Scenario Simulations
# ----------------------
with tabs[1]:
    st.subheader("Monte Carlo Simulation")
    num_simulations = 50
    sim_years = st.number_input("Simulation Years", value=1)
    sim_return = st.number_input("Expected Annual Return (%)", value=8.0)
    sim_volatility = st.number_input("Volatility (%)", value=15.0)

    simulated_paths = []
    for _ in range(num_simulations):
        path = [capital]
        for _ in range(sim_years*12):
            monthly_return = np.random.normal(sim_return/12/100, sim_volatility/np.sqrt(12)/100)
            path.append(path[-1]*(1+monthly_return))
        simulated_paths.append(path)
    fig, ax = plt.subplots(figsize=(20,10))
    for path in simulated_paths:
        ax.plot(range(len(path)), path, color="blue", alpha=0.2)
    ax.set_xlabel("Months")
    ax.set_ylabel("Portfolio Value")
    st.pyplot(fig)

    # Compute expected portfolio value at the end
    final_values = [path[-1] for path in simulated_paths]
    expected_value = np.mean(final_values)
    st.write(f"**Expected Portfolio Value after {sim_years} year(s): ${expected_value:,.2f}**")

    st.subheader('"What If" Scenarios')
    growth_rates = [5, 10, 15]
    what_if_rows = []
    for rate in growth_rates:
        future_value = capital * (1 + rate/100)**sim_years
        what_if_rows.append({"Growth %": rate, "Future Value": future_value})
    what_if_table = pd.DataFrame(what_if_rows)
    st.table(what_if_table)

# # ----------------------
# # 6️⃣ Tips & Insights
# # ----------------------
# with tabs[4]:
#     st.subheader("Tip of the Day")
#     tips = [
#         "Always risk <= 2% of your capital per trade.",
#         "Diversify across asset classes to reduce risk.",
#         "Backtest strategies before trading real capital.",
#         "Keep a detailed trade journal for review."
#     ]
#     st.write(np.random.choice(tips))
