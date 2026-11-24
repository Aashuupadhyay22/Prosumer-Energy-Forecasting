
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_FILE = PROJECT_ROOT / "predictions" / "test_predictions.csv"

SAMPLE_FILE = (
    PROJECT_ROOT
    / "Data"
    / "example_test_files"
    / "sample_submission.csv"
)

# Check files exist
if not PREDICTION_FILE.exists():
    raise FileNotFoundError(f"Prediction file not found: {PREDICTION_FILE}")

if not SAMPLE_FILE.exists():
    raise FileNotFoundError(f"Sample submission not found: {SAMPLE_FILE}")

# Load files
predictions = pd.read_csv(PREDICTION_FILE)
sample = pd.read_csv(SAMPLE_FILE)

print("=" * 55)
print("FINAL SUBMISSION VALIDATION")
print("=" * 55)

# Display shapes and columns
print("\nPrediction file shape:", predictions.shape)
print("Sample submission shape:", sample.shape)

print("\nPrediction columns:", predictions.columns.tolist())
print("Sample columns:", sample.columns.tolist())

# Check column names and order
print("\nColumn names match:",
      predictions.columns.tolist() == sample.columns.tolist())

# Check row counts
print("Row counts match:", len(predictions) == len(sample))

# Check missing values
print("\nMissing prediction values:")
print(predictions.isnull().sum())

# Check infinite values in target
if "target" in predictions.columns:
    print("\nInfinite target values:",
          (~pd.Series(
              predictions["target"].astype(float)
          ).map(lambda x: pd.notna(x) and abs(x) != float("inf"))).sum())

# Check row IDs
if "row_id" in predictions.columns and "row_id" in sample.columns:
    print("\nRow IDs match in order:",
          predictions["row_id"].equals(sample["row_id"]))

# Final validation
valid = (
    predictions.columns.tolist() == sample.columns.tolist()
    and len(predictions) == len(sample)
    and predictions.notna().all().all()
)

if valid and "row_id" in predictions.columns and "row_id" in sample.columns:
    valid = predictions["row_id"].equals(sample["row_id"])

print("\n" + "=" * 55)

if valid:
    print("SUBMISSION FORMAT VALIDATION PASSED!")
else:
    print("SUBMISSION FORMAT NEEDS CORRECTION.")

print("=" * 55)
