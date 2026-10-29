"""
Logistics Data Analytics Internship — end-to-end demo
Creates simulated data, cleans it, plots EDA charts, and compares delivery-time regression models.
All records and resulting metrics are simulated for educational use only.
Install: pip install pandas numpy matplotlib seaborn scikit-learn
Run: python logistics_internship_demo.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rng = np.random.default_rng(42)
n = 1000
distance = rng.uniform(5, 800, n).round(1)
weight = rng.uniform(0.2, 45, n).round(1)
mode = rng.choice(["road", "rail", "air"], n, p=[0.68, 0.20, 0.12])
traffic = rng.choice(["low", "medium", "high"], n, p=[0.30, 0.45, 0.25])
weather = rng.choice(["clear", "rain", "storm"], n, p=[0.68, 0.27, 0.05])
priority = rng.choice(["standard", "express"], n, p=[0.75, 0.25])
mode_factor = pd.Series(mode).map({"road": 1.0, "rail": 0.75, "air": 0.35}).to_numpy()
traffic_add = pd.Series(traffic).map({"low": 0.5, "medium": 2.0, "high": 5.0}).to_numpy()
weather_add = pd.Series(weather).map({"clear": 0.0, "rain": 2.0, "storm": 6.0}).to_numpy()
priority_add = pd.Series(priority).map({"standard": 2.0, "express": -1.0}).to_numpy()
actual = np.maximum(1, distance / (45 / mode_factor) + traffic_add + weather_add + weight * 0.06 + priority_add + rng.normal(0, 2, n))
planned = np.maximum(1, actual + rng.normal(1.5, 3.5, n))
cost = np.maximum(distance * pd.Series(mode).map({"road": 1.8, "rail": 1.25, "air": 4.8}).to_numpy() + weight * 2.1 + rng.normal(0, 18, n), 5).round(2)

df = pd.DataFrame({
    "shipment_id": [f"S{i:05d}" for i in range(n)],
    "shipment_date": pd.date_range("2026-01-01", periods=n, freq="D"),
    "distance_km": distance, "weight_kg": weight,
    "transport_mode": mode, "traffic_level": traffic, "weather": weather,
    "delivery_priority": priority, "planned_delivery_hours": planned.round(2),
    "actual_delivery_hours": actual.round(2), "transport_cost": cost
})
for col in ["distance_km", "weight_kg", "weather"]:
    df.loc[rng.choice(df.index, 12, replace=False), col] = np.nan
df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
df.to_csv("logistics_shipments_raw.csv", index=False)
print("Raw shape:", df.shape)
print("Missing values before cleaning:\\n", df.isna().sum())

df.columns = df.columns.str.strip().str.lower()
df["shipment_date"] = pd.to_datetime(df["shipment_date"], errors="coerce")
for col in ["transport_mode", "traffic_level", "weather", "delivery_priority"]:
    df[col] = df[col].astype("string").str.strip().str.lower()
df = df.drop_duplicates(subset=["shipment_id"], keep="last")
numeric_cols = ["distance_km", "weight_kg", "planned_delivery_hours", "actual_delivery_hours", "transport_cost"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df[col] <= 0, col] = np.nan
    df[col] = df[col].fillna(df[col].median())
df["weather"] = df["weather"].fillna("unknown")
df["on_time"] = df["actual_delivery_hours"] <= df["planned_delivery_hours"]
df["delay_hours"] = df["actual_delivery_hours"] - df["planned_delivery_hours"]
df["cost_per_km"] = df["transport_cost"] / df["distance_km"].clip(lower=1)
df.to_csv("logistics_shipments_clean.csv", index=False)
print("Clean shape:", df.shape)
print("On-time delivery rate (%):", round(df["on_time"].mean() * 100, 2))
print("Average delivery time (hours):", round(df["actual_delivery_hours"].mean(), 2))
print("Average shipment cost:", round(df["transport_cost"].mean(), 2))
print("Cleaned missing values:\\n", df.isna().sum())

sns.set_theme()
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="actual_delivery_hours", kde=True, color="steelblue")
plt.title("Distribution of Actual Delivery Time")
plt.tight_layout()
plt.savefig("week3_delivery_time_histogram.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="transport_mode", y="on_time", errorbar=None, color="steelblue")
plt.title("On-Time Delivery Rate by Transport Mode")
plt.ylabel("On-time rate")
plt.tight_layout()
plt.savefig("week3_ontime_by_mode.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="distance_km", y="actual_delivery_hours", hue="traffic_level", alpha=0.7, palette="deep")
plt.title("Distance vs Actual Delivery Time")
plt.tight_layout()
plt.savefig("week3_distance_vs_time.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 6))
sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Numeric Variables")
plt.tight_layout()
plt.savefig("week3_correlation_heatmap.png", dpi=150)
plt.close()

target = "actual_delivery_hours"
features = ["distance_km", "weight_kg", "transport_mode", "traffic_level", "weather", "delivery_priority"]
X, y = df[features], df[target]
num = ["distance_km", "weight_kg"]
cat = ["transport_mode", "traffic_level", "weather", "delivery_priority"]
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    "Mean baseline": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, min_samples_leaf=2)
}
results = []
for name, model in models.items():
    pipe = Pipeline([("preprocess", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results.append({"model": name, "MAE_hours": mean_absolute_error(y_test, pred),
                    "RMSE_hours": np.sqrt(mean_squared_error(y_test, pred)), "R2": r2_score(y_test, pred)})
results = pd.DataFrame(results).sort_values("MAE_hours")
results.to_csv("week4_model_metrics.csv", index=False)
print("\\nModel comparison (simulated data):\\n", results.to_string(index=False))
print("\\nFiles created: logistics_shipments_raw.csv, logistics_shipments_clean.csv, 4 chart PNGs, week4_model_metrics.csv")
