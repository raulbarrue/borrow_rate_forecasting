"""
data_loader.py
-------------------------------------
Fetches market data from Yahoo Finance and generates a synthetic borrow-rate dataset
for Prime Finance optimization prototype.

Author: Raul Barrue
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf


def fetch_market_data(
    tickers=None, period="1y", interval="1d", save_path="data/market_data.csv"
):
    """
    Fetch daily market data (Close, Volume) for given tickers and save to CSV.
    """
    if tickers is None:
        tickers = ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC"]

    print(f"Fetching {len(tickers)} tickers from Yahoo Finance...")
    data = yf.download(tickers, period=period, interval=interval, auto_adjust=True)
    data = data[["Close", "Volume"]]

    # Flatten multi-index: ('Close', 'JPM') → 'JPM_Close'
    data.columns = [f"{col[1]}_{col[0]}" for col in data.columns]
    data.reset_index(inplace=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data.to_csv(save_path, index=False)
    print(f"Market data saved → {save_path}")

    return data


def generate_synthetic_borrow_rates(
    market_data_path="data/market_data.csv", save_path="data/synthetic_borrow_rates.csv"
):
    """
    Generate synthetic borrow-rate data using volatility and momentum proxies.
    Formula:
        rate = 0.05 * rolling_volatility + 0.03 * momentum + noise
    """
    print("Generating synthetic borrow rates...")
    df = pd.read_csv(market_data_path, parse_dates=["Date"])
    tickers = sorted({col.split("_")[0] for col in df.columns if "_Close" in col})

    synthetic_rows = []
    for t in tickers:
        close = df[f"{t}_Close"]
        vol_rolling = close.pct_change().rolling(10).std()
        momentum = close.pct_change(5)
        noise = np.random.normal(0, 0.002, len(close))
        synthetic_rate = 0.05 * vol_rolling + 0.03 * momentum + noise

        temp = pd.DataFrame({
            "Date": df["Date"],
            "Ticker": t,
            "SyntheticBorrowRate": synthetic_rate
        })
        synthetic_rows.append(temp)

    rates = pd.concat(synthetic_rows)
    rates.sort_values(["Ticker", "Date"], inplace=True)
    rates.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rates.to_csv(save_path, index=False)
    print(f"Synthetic borrow rates saved → {save_path}")

    return rates


if __name__ == "__main__":
    market_df = fetch_market_data()
    rates_df = generate_synthetic_borrow_rates()
