
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Processed_Data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "evaluation_results"

RESULT_DIR.mkdir(exist_ok=True)

# ============================================================
# 2. LOAD DATA AND MODEL
# ============================================================

print("Loading dataset and model...")

df = pd.read_csv(
    DATA_DIR / "feature_engineered_training.csv"
)

df["datetime"] = pd.to_datetime(df["datetime"])

model = joblib.load(
    MODEL_DIR / "energy_forecasting_model.pkl"
)

with open(MODEL_DIR / "model_features.json") as file:
    feature_columns = json.load(file)

# IMPORTANT: Sort before selecting features.
# This reproduces the order used during model training.

df = df.sort_values(
    "datetime"
).reset_index(drop=True)

# ============================================================
# 3. PREPARE FEATURES
# ============================================================

X = df[
    [col for col in feature_columns if col in df.columns]
].select_dtypes(include=[np.number]).copy()

X = X.replace([np.inf, -np.inf], np.nan)

y = df["target"]

print("Dataset shape:", df.shape)
print("Feature shape:", X.shape)

# ============================================================
# 4. RECREATE THE SAME VALIDATION SPLIT
# ============================================================

split_position = int(len(df) * 0.80)

X_valid = X.iloc[split_position:]
y_valid = y.iloc[split_position:]

validation_dates = df["datetime"].iloc[split_position:]

print("Validation rows:", len(X_valid))

# ============================================================
# 5. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(X_valid)

# ============================================================
# 6. EVALUATION METRICS
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
# 7. ACTUAL VS PREDICTED DATA
# ============================================================

comparison = pd.DataFrame({
    "datetime": validation_dates.values,
    "actual_target": y_valid.values,
    "predicted_target": predictions
})

comparison["prediction_error"] = (
    comparison["actual_target"]
    - comparison["predicted_target"]
)

comparison["absolute_error"] = (
    comparison["prediction_error"].abs()
)

comparison.to_csv(
    RESULT_DIR / "actual_vs_predicted.csv",
    index=False
)

# ============================================================
# 8. ACTUAL VS PREDICTED LINE GRAPH
# ============================================================

plot_data = comparison.sample(
    n=min(2000, len(comparison)),
    random_state=42
).sort_values("datetime")

plt.figure(figsize=(12, 6))

plt.plot(
    plot_data["datetime"],
    plot_data["actual_target"],
    label="Actual"
)

plt.plot(
    plot_data["datetime"],
    plot_data["predicted_target"],
    label="Predicted"
)

plt.title("Actual vs Predicted Energy Values")
plt.xlabel("Date")
plt.ylabel("Target")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.close()

# ============================================================
# 9. SCATTER PLOT
# ============================================================

scatter_data = comparison.sample(
    n=min(5000, len(comparison)),
    random_state=42
)

plt.figure(figsize=(8, 8))

plt.scatter(
    scatter_data["actual_target"],
    scatter_data["predicted_target"],
    alpha=0.3
)

min_value = min(
    scatter_data["actual_target"].min(),
    scatter_data["predicted_target"].min()
)

max_value = max(
    scatter_data["actual_target"].max(),
    scatter_data["predicted_target"].max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title("Actual vs Predicted Scatter Plot")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "prediction_scatter.png",
    dpi=300
)

plt.close()

# ============================================================
# 10. RESIDUAL DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    comparison["prediction_error"],
    bins=50,
    edgecolor="black"
)

plt.title("Prediction Error Distribution")
plt.xlabel("Actual - Predicted")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "residual_distribution.png",
    dpi=300
)

plt.close()

# ============================================================
# 11. SAVE EVALUATION REPORT
# ============================================================

results = {
    "model": "HistGradientBoostingRegressor",
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2),
    "validation_rows": int(len(y_valid))
}

with open(
    RESULT_DIR / "evaluation_report.json",
    "w"
) as file:
    json.dump(results, file, indent=4)

# ============================================================
# 12. COMPLETION
# ============================================================

print("\nEVALUATION COMPLETED")
print("=" * 50)

print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2:", round(r2, 4))

print("\nResults saved in:", RESULT_DIR)
