
# ============================================================
# PROSUMER ENERGY FORECASTING
# MODEL TRAINING AND VALIDATION
# ============================================================

import json
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Processed_Data"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "feature_engineered_training.csv"

print("=" * 65)
print("PROSUMER ENERGY FORECASTING - MODEL TRAINING")
print("=" * 65)

# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["datetime"] = pd.to_datetime(df["datetime"])

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

# ============================================================
# 3. SORT DATA CHRONOLOGICALLY
# ============================================================

df = df.sort_values("datetime").reset_index(drop=True)

# ============================================================
# 4. SELECT MODEL FEATURES
# ============================================================

# Remove target and columns that are identifiers,
# timestamps, or merge bookkeeping indicators.

excluded_columns = [
    "target",
    "datetime",
    "row_id",
    "prediction_unit_id",
    "electricity_merge",
    "gas_merge"
]

feature_columns = [
    col for col in df.columns
    if col not in excluded_columns
]

# Keep numeric features only.
X = df[feature_columns].select_dtypes(
    include=[np.number]
).copy()

y = df["target"]

# Replace infinite values with missing values.
X = X.replace([np.inf, -np.inf], np.nan)

print("\nNumber of model features:", X.shape[1])
print("Feature columns:")
print(list(X.columns))

# ============================================================
# 5. CHRONOLOGICAL TRAIN-VALIDATION SPLIT
# ============================================================

print("\nCREATING CHRONOLOGICAL TRAIN-VALIDATION SPLIT")

split_position = int(len(df) * 0.80)

X_train = X.iloc[:split_position]
X_valid = X.iloc[split_position:]

y_train = y.iloc[:split_position]
y_valid = y.iloc[split_position:]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_valid))

print(
    "Training period:",
    df["datetime"].iloc[0],
    "to",
    df["datetime"].iloc[split_position - 1]
)

print(
    "Validation period:",
    df["datetime"].iloc[split_position],
    "to",
    df["datetime"].iloc[-1]
)

# ============================================================
# 6. CREATE MODEL
# ============================================================

print("\nINITIALIZING HISTOGRAM GRADIENT BOOSTING MODEL")

model = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    early_stopping=True,
    random_state=42
)

# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\nMODEL TRAINING STARTED...")
print("Please wait. Training may take some time.")

model.fit(X_train, y_train)

print("\nMODEL TRAINING COMPLETED!")

# ============================================================
# 8. MAKE VALIDATION PREDICTIONS
# ============================================================

print("\nGENERATING VALIDATION PREDICTIONS")

predictions = model.predict(X_valid)

# ============================================================
# 9. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_valid, predictions)

rmse = np.sqrt(
    mean_squared_error(y_valid, predictions)
)

r2 = r2_score(y_valid, predictions)

print("\nMODEL PERFORMANCE")
print("=" * 50)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# ============================================================
# 10. SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "energy_forecasting_model.pkl"

joblib.dump(model, model_path)

print("\nTrained model saved:")
print(model_path)

# ============================================================
# 11. SAVE FEATURE COLUMNS
# ============================================================

feature_path = MODEL_DIR / "model_features.json"

with open(feature_path, "w") as file:
    json.dump(feature_columns, file, indent=4)

print("\nModel feature list saved:")
print(feature_path)

# ============================================================
# 12. SAVE EVALUATION RESULTS
# ============================================================

results = {
    "model": "HistGradientBoostingRegressor",
    "training_rows": int(len(X_train)),
    "validation_rows": int(len(X_valid)),
    "number_of_features": int(X.shape[1]),
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2)
}

results_path = MODEL_DIR / "model_evaluation.json"

with open(results_path, "w") as file:
    json.dump(results, file, indent=4)

print("\nEvaluation results saved:")
print(results_path)

# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("MODEL TRAINING AND EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nFinal Model Performance:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

print("\nAll model files saved in:", MODEL_DIR)
