
# ============================================================
# PROSUMER ENERGY FORECASTING
# TEST PREDICTION PIPELINE
# ============================================================

import json
import joblib
import pandas as pd
import numpy as np

from pathlib import Path

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "predictions"

TEST_DIR = DATA_DIR / "example_test_files"

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("PROSUMER ENERGY FORECASTING - TEST PREDICTIONS")
print("=" * 65)

# ============================================================
# 2. LOAD FILES
# ============================================================

test_file = TEST_DIR / "test.csv"
train_file = DATA_DIR / "train.csv"
client_file = DATA_DIR / "client.csv"
electricity_file = DATA_DIR / "electricity_prices.csv"
gas_file = DATA_DIR / "gas_prices.csv"
revealed_file = TEST_DIR / "revealed_targets.csv"

model_file = MODEL_DIR / "energy_forecasting_model.pkl"
features_file = MODEL_DIR / "model_features.json"

if not test_file.exists():
    raise FileNotFoundError(f"Test file not found: {test_file}")

if not train_file.exists():
    raise FileNotFoundError(f"Training file not found: {train_file}")

if not model_file.exists():
    raise FileNotFoundError(f"Model not found: {model_file}")

test = pd.read_csv(test_file)

model = joblib.load(model_file)

with open(features_file, "r") as file:
    feature_columns = json.load(file)

print("\nTest dataset loaded:", test.shape)
print("Test columns:", list(test.columns))
print("Model loaded successfully!")

# ============================================================
# 3. STANDARDIZE DATETIME COLUMN
# ============================================================

def standardize_datetime(data):

    data = data.copy()

    possible_names = [
        "datetime",
        "prediction_datetime",
        "forecast_date",
        "date"
    ]

    found_column = None

    for column in possible_names:
        if column in data.columns:
            found_column = column
            break

    if found_column is None:
        raise ValueError(
            "No datetime column found. Available columns: "
            + str(list(data.columns))
        )

    if found_column != "datetime":
        data = data.rename(
            columns={found_column: "datetime"}
        )

    data["datetime"] = pd.to_datetime(
        data["datetime"],
        errors="coerce"
    )

    if data["datetime"].isna().any():
        raise ValueError(
            "Some dates could not be converted. "
            "Check the datetime column in the input file."
        )

    return data


test = standardize_datetime(test)

# Preserve original test order.
test["_original_order"] = np.arange(len(test))

print("\nTest datetime column standardized.")

# ============================================================
# 4. LOAD HISTORICAL TARGET DATA
# ============================================================

history_frames = []

train_history = pd.read_csv(train_file)

train_history = standardize_datetime(train_history)

required_columns = [
    "county",
    "is_business",
    "product_type",
    "is_consumption",
    "datetime",
    "target"
]

missing_columns = [
    col for col in required_columns
    if col not in train_history.columns
]

if missing_columns:
    raise ValueError(
        "Training history is missing columns: "
        + str(missing_columns)
    )

history_frames.append(
    train_history[required_columns].copy()
)

# Load revealed targets if available.
if revealed_file.exists():

    revealed = pd.read_csv(revealed_file)

    try:
        revealed = standardize_datetime(revealed)

        if all(
            col in revealed.columns
            for col in required_columns
        ):
            history_frames.append(
                revealed[required_columns].copy()
            )

            print("Revealed historical targets loaded.")

    except ValueError:
        print(
            "Warning: Revealed targets do not have "
            "a usable datetime column. Skipping them."
        )

history = pd.concat(
    history_frames,
    ignore_index=True
)

history = history.dropna(
    subset=["datetime", "target"]
)

group_columns = [
    "county",
    "is_business",
    "product_type",
    "is_consumption"
]

history = history.drop_duplicates(
    subset=group_columns + ["datetime"],
    keep="last"
)

print("\nHistorical records:", len(history))

# ============================================================
# 5. CALENDAR FEATURES
# ============================================================

def add_calendar_features(data):

    data = data.copy()

    dt = data["datetime"]

    data["year"] = dt.dt.year
    data["month"] = dt.dt.month
    data["day"] = dt.dt.day
    data["hour"] = dt.dt.hour
    data["day_of_week"] = dt.dt.dayofweek
    data["day_of_year"] = dt.dt.dayofyear

    data["week_of_year"] = (
        dt.dt.isocalendar().week.astype(float)
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["is_month_start"] = (
        dt.dt.is_month_start
    ).astype(int)

    data["is_month_end"] = (
        dt.dt.is_month_end
    ).astype(int)

    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour"] / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour"] / 24
    )

    data["day_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["day_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    data["business_consumption"] = (
        data["is_business"] * data["is_consumption"]
    )

    data["product_business"] = (
        data["product_type"] * data["is_business"]
    )

    return data


test = add_calendar_features(test)

# ============================================================
# 6. CREATE HISTORICAL LAG FEATURES
# ============================================================

print("\nCreating historical lag features...")

test_history = test.copy()
test_history["target"] = np.nan

combined = pd.concat(
    [history, test_history],
    ignore_index=True,
    sort=False
)

combined = combined.sort_values(
    group_columns + ["datetime"]
).reset_index(drop=True)

grouped_target = combined.groupby(
    group_columns,
    sort=False
)["target"]

combined["target_lag_1"] = grouped_target.shift(1)
combined["target_lag_24"] = grouped_target.shift(24)
combined["target_lag_48"] = grouped_target.shift(48)
combined["target_lag_168"] = grouped_target.shift(168)

combined["target_rolling_mean_24"] = (
    combined.groupby(group_columns)["target"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=24,
            min_periods=24
        ).mean()
    )
)

combined["target_rolling_mean_168"] = (
    combined.groupby(group_columns)["target"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=168,
            min_periods=168
        ).mean()
    )
)

combined["target_rolling_std_24"] = (
    combined.groupby(group_columns)["target"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=24,
            min_periods=24
        ).std()
    )
)

# Extract only original test rows.
test_features = combined[
    combined["_original_order"].notna()
].copy()

test_features = test_features.sort_values(
    "_original_order"
).reset_index(drop=True)

print("Lag features created.")

# ============================================================
# 7. MERGE CLIENT DATA
# ============================================================

print("\nMerging client data...")

if client_file.exists():

    client = pd.read_csv(client_file)

    merge_keys = [
        "county",
        "is_business",
        "product_type",
        "data_block_id"
    ]

    if all(
        col in client.columns and col in test_features.columns
        for col in merge_keys
    ):

        client_columns = merge_keys + [
            col for col in [
                "eic_count",
                "installed_capacity"
            ]
            if col in client.columns
        ]

        client = client[client_columns].drop_duplicates(
            subset=merge_keys,
            keep="last"
        )

        test_features = test_features.merge(
            client,
            on=merge_keys,
            how="left",
            validate="many_to_one"
        )

# ============================================================
# 8. MERGE ELECTRICITY PRICES
# ============================================================

print("\nMerging electricity prices...")

if electricity_file.exists():

    electricity = pd.read_csv(electricity_file)

    electricity = standardize_datetime(electricity)

    electricity = electricity[
        ["datetime", "euros_per_mwh"]
    ].drop_duplicates(
        subset=["datetime"],
        keep="last"
    )

    test_features = test_features.merge(
        electricity,
        on="datetime",
        how="left",
        validate="many_to_one"
    )

# ============================================================
# 9. MERGE GAS PRICES
# ============================================================

print("\nMerging gas prices...")

if gas_file.exists():

    gas = pd.read_csv(gas_file)

    gas_columns = [
        "data_block_id",
        "lowest_price_per_mwh",
        "highest_price_per_mwh"
    ]

    if all(col in gas.columns for col in gas_columns):

        gas = gas[gas_columns].drop_duplicates(
            subset=["data_block_id"],
            keep="last"
        )

        test_features = test_features.merge(
            gas,
            on="data_block_id",
            how="left",
            validate="many_to_one"
        )


# ============================================================
# 10. PREPARE EXACT MODEL INPUT FEATURES
# ============================================================

print("\nPreparing exact model features...")

# Use the exact feature names stored by the trained model.
required_features = list(model.feature_names_in_)

print("Required model features:", len(required_features))

# Remove accidental merge columns and other extra columns
# by selecting only the features used during model training.
for column in required_features:
    if column not in test_features.columns:
        test_features[column] = np.nan

X_test = test_features[required_features].copy()

# Convert every feature to numeric.
X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce"
)

# Replace infinite values.
X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

# Verify exact feature names and order.
assert list(X_test.columns) == required_features

print("Model input shape:", X_test.shape)
print("Feature names match the trained model!")

# ============================================================
# 11. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(X_test)

# ============================================================
# 12. SAVE PREDICTIONS
# ============================================================


# Create submission with the exact required columns

required_columns = ["row_id", "data_block_id"]

missing_columns = [
    col for col in required_columns
    if col not in test.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required test columns: {missing_columns}"
    )

submission = test[required_columns].copy()

submission["target"] = predictions

# Match sample submission column order
submission = submission[
    ["row_id", "data_block_id", "target"]
]


output_file = OUTPUT_DIR / "test_predictions.csv"

submission.to_csv(
    output_file,
    index=False
)

# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("TEST PREDICTIONS COMPLETED")
print("=" * 65)

print("Test rows:", len(test))
print("Predictions generated:", len(predictions))
print("Missing predictions:", submission["target"].isna().sum())

print("Minimum prediction:", round(float(predictions.min()), 4))
print("Maximum prediction:", round(float(predictions.max()), 4))
print("Mean prediction:", round(float(predictions.mean()), 4))

print("\nFirst five predictions:")
print(submission.head())

print("\nSaved file:")
print(output_file)

print("\nPROCESS COMPLETED SUCCESSFULLY!")


