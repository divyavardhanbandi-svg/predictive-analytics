"""
Predictive Analytics - Sales Demand Forecasting
------------------------------------------------
Simple base project:
  1. Generate (or load) daily sales data
  2. Feature engineering (date + lag features)
  3. Train Linear Regression and Random Forest models
  4. Evaluate with MAE, RMSE and R2
  5. Plot actual vs predicted sales
  6. Forecast the next 30 days

Install:  pip install pandas numpy scikit-learn matplotlib
Run:      python predictive_analytics.py
To use your own data, replace load_data() with pd.read_csv("your_file.csv")
(it needs a 'date' column and a 'sales' column).
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # works without a display; remove if you want pop-up plots
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------
def load_data(n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    """Create synthetic daily sales with trend, weekly + yearly seasonality and noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    trend = 200 + 0.25 * t
    weekly = 30 * np.where(dates.dayofweek >= 5, 1, 0)          # weekend boost
    yearly = 40 * np.sin(2 * np.pi * dates.dayofyear / 365.25)   # seasonal wave
    noise = rng.normal(0, 12, n_days)

    sales = trend + weekly + yearly + noise
    return pd.DataFrame({"date": dates, "sales": sales.round(2)})


# ----------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["time_index"] = np.arange(len(df))
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["rolling_mean_7"] = df["sales"].shift(1).rolling(7).mean()
    return df.dropna().reset_index(drop=True)


FEATURES = [
    "day_of_week", "month", "day_of_year", "is_weekend",
    "time_index", "lag_1", "lag_7", "rolling_mean_7",
]


# ----------------------------------------------------------------------
# 3. Train / evaluate
# ----------------------------------------------------------------------
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<18} MAE: {mae:8.2f} | RMSE: {rmse:8.2f} | R2: {r2:6.3f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    raw = load_data()
    df = add_lag_features(add_calendar_features(raw))

    # Time-based split (never shuffle time series): last 20% is the test set
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    X_train, y_train = train[FEATURES], train["sales"]
    X_test, y_test = test[FEATURES], test["sales"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }

    print(f"Train rows: {len(train)} | Test rows: {len(test)}\n")
    results, preds = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds[name] = model.predict(X_test)
        results.append(evaluate(name, y_test, preds[name]))

    results_df = pd.DataFrame(results).round(3)
    results_df.to_csv("model_results.csv", index=False)
    best_name = results_df.sort_values("RMSE").iloc[0]["model"]
    print(f"\nBest model: {best_name}")

    # ------------------------------------------------------------------
    # 4. Plot actual vs predicted
    # ------------------------------------------------------------------
    plt.figure(figsize=(11, 5))
    plt.plot(test["date"], y_test, label="Actual", color="black", linewidth=1.5)
    for name, p in preds.items():
        plt.plot(test["date"], p, label=name, alpha=0.8)
    plt.title("Sales Demand: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # 5. Forecast next 30 days with the best model (recursive forecasting)
    # ------------------------------------------------------------------
    best_model = models[best_name]
    history = raw.copy()
    future_rows = []

    for _ in range(30):
        next_date = history["date"].iloc[-1] + pd.Timedelta(days=1)
        history = pd.concat(
            [history, pd.DataFrame({"date": [next_date], "sales": [np.nan]})],
            ignore_index=True,
        )
        feat = add_calendar_features(history)
        feat["lag_1"] = feat["sales"].shift(1)
        feat["lag_7"] = feat["sales"].shift(7)
        feat["rolling_mean_7"] = feat["sales"].shift(1).rolling(7).mean()

        row = feat.iloc[[-1]][FEATURES]
        pred = float(best_model.predict(row)[0])
        history.loc[history.index[-1], "sales"] = pred
        future_rows.append({"date": next_date, "forecast_sales": round(pred, 2)})

    forecast = pd.DataFrame(future_rows)
    forecast.to_csv("forecast_next_30_days.csv", index=False)
    print("\nNext 7 days forecast:")
    print(forecast.head(7).to_string(index=False))

    plt.figure(figsize=(11, 5))
    plt.plot(raw["date"].tail(90), raw["sales"].tail(90), label="Last 90 days (actual)")
    plt.plot(forecast["date"], forecast["forecast_sales"], label="30-day forecast", color="red")
    plt.title(f"30-Day Sales Forecast ({best_name})")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig("forecast_30_days.png", dpi=150)
    plt.close()

    print("\nSaved: model_results.csv, forecast_next_30_days.csv, "
          "actual_vs_predicted.png, forecast_30_days.png")


if __name__ == "__main__":
    main()
