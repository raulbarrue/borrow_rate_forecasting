"""
model_train.py
-------------------------------------
Trains a simple Gradient Boosting Regressor to predict next-day synthetic borrow rates.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RateLag1"] = df.groupby("Ticker")["SyntheticBorrowRate"].shift(1)
    df["RateLag3"] = df.groupby("Ticker")["SyntheticBorrowRate"].shift(3)
    df["RateLag5"] = df.groupby("Ticker")["SyntheticBorrowRate"].shift(5)
    df["RollingVol"] = df.groupby("Ticker")["SyntheticBorrowRate"].transform(lambda x: x.rolling(10).std())
    df["Momentum"] = df.groupby("Ticker")["SyntheticBorrowRate"].transform(lambda x: x - x.shift(5))
    df.dropna(inplace=True)
    return df


def train_model(data_path="data/synthetic_borrow_rates.csv", save_path="models/borrow_rate_model.pkl"):
    print("Training Gradient Boosting Model...")
    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = prepare_features(df)

    X = df[["RateLag1", "RateLag3", "RateLag5", "RollingVol", "Momentum"]]
    y = df["SyntheticBorrowRate"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print(f"RMSE: {rmse:.6f}")
    print(f"R²:   {r2:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Model saved → {save_path}")

    return model


if __name__ == "__main__":
    train_model()
