
# ============================================================
# PROSUMER ENERGY FORECASTING
# FEATURE ENGINEERING
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = PROJECT_ROOT / "Processed_Data"

OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "processed_training_data.csv"

print("=" * 65)
print("PROSUMER ENERGY FORECASTING - FEATURE ENGINEERING")
print("=" * 65)

# ============================================================
# 2. LOAD PROCESSED DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully!")
print("Original shape:", df.shape)

# ============================================================
# 3. DATETIME CONVERSION
# ============================================================

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)

df = df.dropna(subset=["datetime"]).copy()

# ============================================================
# 4. SORT DATA
# ============================================================

group_columns = [
    "county",
    "is_business",
    "product_type",
    "is_consumption"
]

df = df.sort_values(
    group_columns + ["datetime"]
).reset_index(drop=True)

# ============================================================
# 5. CALENDAR FEATURES
# ============================================================

print("\nCREATING CALENDAR FEATURES")

df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
df["day_of_year"] = df["datetime"].dt.dayofyear
df["week_of_year"] = (
    df["datetime"].dt.isocalendar().week.astype(int)
)

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

df["is_month_start"] = (
    df["datetime"].dt.is_month_start
).astype(int)

df["is_month_end"] = (
    df["datetime"].dt.is_month_end
).astype(int)

# ============================================================
# 6. CYCLICAL TIME FEATURES
# ============================================================

print("\nCREATING CYCLICAL FEATURES")

df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

df["day_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["day_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)

df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)

# ============================================================
# 7. BUSINESS AND CONSUMPTION FEATURES
# ============================================================

print("\nCREATING BUSINESS FEATURES")

df["business_consumption"] = (
    df["is_business"] * df["is_consumption"]
)

df["product_business"] = (
    df["product_type"] * df["is_business"]
)

# ============================================================
# 8. LAG FEATURES
# ============================================================

print("\nCREATING LAG FEATURES")

# Each group represents a separate prosumer series.
# Shift ensures only previous observations are used.

grouped_target = df.groupby(
    group_columns,
    sort=False
)["target"]

df["target_lag_1"] = grouped_target.shift(1)
df["target_lag_24"] = grouped_target.shift(24)
df["target_lag_48"] = grouped_target.shift(48)
df["target_lag_168"] = grouped_target.shift(168)

# ============================================================
# 9. ROLLING FEATURES
# ============================================================

print("\nCREATING ROLLING FEATURES")

df["target_rolling_mean_24"] = (
    df.groupby(group_columns, sort=False)["target"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=24,
            min_periods=24
        ).mean()
    )
)

df["target_rolling_mean_168"] = (
    df.groupby(group_columns, sort=False)["target"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=168,
            min_periods=168
        ).mean()
    )
)

df["target_rolling_std_24"] = (
    df.groupby(group_columns, sort=False)["target"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=24,
            min_periods=24
        ).std()
    )
)

# ============================================================
# 10. CHECK MISSING VALUES
# ============================================================

print("\nFEATURE ENGINEERING MISSING VALUE CHECK")

print(df.isnull().sum())

print("\nMissing value percentages:")

print(
    (df.isnull().mean() * 100).round(2)
)

# ============================================================
# 11. SAVE FEATURE ENGINEERED DATA
# ============================================================

OUTPUT_FILE = (
    OUTPUT_DIR / "feature_engineered_training.csv"
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFEATURE ENGINEERING COMPLETED")
print("=" * 65)

print("Original training rows:", len(df))
print("Final dataset shape:", df.shape)
print("Total features:", len(df.columns))

print("\nSaved file:")
print(OUTPUT_FILE)

print("\nFeature engineering finished successfully!")
