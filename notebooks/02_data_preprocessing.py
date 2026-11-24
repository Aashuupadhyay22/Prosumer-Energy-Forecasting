
# ============================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================
# 2. SET PROJECT AND DATA PATHS
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"

print("Project folder:", PROJECT_ROOT)
print("Data folder:", DATA_DIR)

# ============================================
# 3. LOAD TRAINING DATASET
# ============================================

train_df = pd.read_csv(DATA_DIR / "train.csv")

print("\nTraining dataset loaded successfully!")
print("Shape:", train_df.shape)

print("\nFirst 5 rows:")
print(train_df.head())

print("\nColumn names:")
print(train_df.columns.tolist())

print("\nData types:")
print(train_df.dtypes)

# ============================================
# 4. CHECK MISSING VALUES AND DUPLICATES
# ============================================

print("\nMissing values in each column:")
print(train_df.isnull().sum())

print("\nTotal missing values:")
print(train_df.isnull().sum().sum())

print("\nTotal duplicate rows:")
print(train_df.duplicated().sum())

print("\nDataset information:")
train_df.info()

# ============================================
# 5. CONVERT DATETIME COLUMN
# ============================================

train_df["datetime"] = pd.to_datetime(
    train_df["datetime"],
    errors="coerce"
)

print("\nDatetime conversion completed!")
print("Datetime data type:", train_df["datetime"].dtype)

print("\nInvalid datetime values:")
print(train_df["datetime"].isnull().sum())


# ============================================
# 6. DETAILED MISSING VALUE ANALYSIS
# ============================================

missing_summary = pd.DataFrame({
    "Missing_Count": train_df.isnull().sum(),
    "Missing_Percentage": (
        train_df.isnull().sum() / len(train_df)
    ) * 100
})

print("\nMissing Value Summary:")
print(missing_summary)

print("\nRows with missing target:")
print(train_df["target"].isnull().sum())


# ============================================
# 7. HANDLE MISSING TARGET VALUES
# ============================================

train_clean = train_df.dropna(subset=["target"]).copy()

print("\nMissing target handling completed!")

print("Original dataset shape:", train_df.shape)
print("Cleaned dataset shape:", train_clean.shape)

print("Missing targets after cleaning:",
      train_clean["target"].isnull().sum())

# ============================================
# 8. VERIFY CLEANED DATASET
# ============================================

print("\nMissing values in cleaned dataset:")
print(train_clean.isnull().sum())

print("\nDuplicate rows in cleaned dataset:")
print(train_clean.duplicated().sum())

print("\nCleaned dataset information:")
train_clean.info()


# ============================================
# 9. CHECK NEGATIVE TARGET VALUES
# ============================================

negative_targets = (train_clean["target"] < 0).sum()

print("\nNegative target values:", negative_targets)

print("\nTarget statistics:")
print(train_clean["target"].describe())


# ============================================
# 10. CHECK ZERO TARGET VALUES
# ============================================

zero_targets = (train_clean["target"] == 0).sum()

zero_percentage = (
    zero_targets / len(train_clean)
) * 100

print("\nZero target values:", zero_targets)
print("Percentage of zero targets:", round(zero_percentage, 2), "%")

# ============================================
# 11. IDENTIFY TARGET OUTLIERS USING IQR
# ============================================

Q1 = train_clean["target"].quantile(0.25)
Q3 = train_clean["target"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = train_clean[
    (train_clean["target"] < lower_bound) |
    (train_clean["target"] > upper_bound)
]

print("\nIQR Outlier Analysis")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)
print("Number of potential outliers:", len(outliers))
print("Percentage of potential outliers:",
      round(len(outliers) / len(train_clean) * 100, 2), "%")



# ============================================
# 12. INSPECT LARGEST TARGET VALUES
# ============================================

largest_targets = train_clean.nlargest(10, "target")

print("\n10 Largest Target Values:")
print(
    largest_targets[
        ["county", "is_business", "product_type",
         "target", "is_consumption", "datetime"]
    ]
)

# ============================================
# 13. TARGET ANALYSIS BY CONSUMPTION TYPE
# ============================================

print("\nTarget statistics by consumption type:")

print(
    train_clean.groupby("is_consumption")["target"]
    .describe()
)

print("\nNumber of records by consumption type:")

print(
    train_clean["is_consumption"].value_counts()
)

# ============================================
# 14. TARGET ANALYSIS BY PRODUCT TYPE
# ============================================

print("\nTarget statistics by product type:")

print(
    train_clean.groupby("product_type")["target"]
    .agg(["count", "mean", "median", "min", "max"])
)

print("\nNumber of records by product type:")

print(train_clean["product_type"].value_counts().sort_index())

# ============================================
# 15. TARGET ANALYSIS BY COUNTY
# ============================================

print("\nNumber of records by county:")
print(train_clean["county"].value_counts().sort_index())

print("\nTarget statistics by county:")
print(
    train_clean.groupby("county")["target"]
    .agg(["count", "mean", "median", "min", "max"])
)

# ============================================
# 15. TARGET ANALYSIS BY COUNTY
# ============================================

print("\nNumber of records by county:")
print(train_clean["county"].value_counts().sort_index())

print("\nTarget statistics by county:")
print(
    train_clean.groupby("county")["target"]
    .agg(["count", "mean", "median", "min", "max"])
)



# ============================================
# 16. TARGET ANALYSIS BY BUSINESS TYPE
# ============================================

print("\nNumber of records by business type:")
print(train_clean["is_business"].value_counts())

print("\nTarget statistics by business type:")

print(
    train_clean.groupby("is_business")["target"]
    .agg(["count", "mean", "median", "min", "max"])
)



# ============================================
# 17. CLIENT DATASET PREPROCESSING
# ============================================

client_df = pd.read_csv(DATA_DIR / "client.csv")

print("\nCLIENT DATASET")
print("=" * 60)

print("Shape:", client_df.shape)

print("\nFirst 5 rows:")
print(client_df.head())

print("\nColumn names:")
print(client_df.columns.tolist())

print("\nData types:")
print(client_df.dtypes)

print("\nMissing values:")
print(client_df.isnull().sum())

print("\nDuplicate rows:")
print(client_df.duplicated().sum())

print("\nDataset information:")
client_df.info()



# ============================================
# 18. CONVERT CLIENT DATE COLUMN
# ============================================

client_df["date"] = pd.to_datetime(
    client_df["date"],
    errors="coerce"
)

print("\nClient date conversion completed!")

print("Date data type:", client_df["date"].dtype)

print(
    "Invalid date values:",
    client_df["date"].isnull().sum()
)

print(
    "Date range:",
    client_df["date"].min(),
    "to",
    client_df["date"].max()
)

# ============================================
# 19. CHECK CLIENT DATASET MERGING KEYS
# ============================================

merge_keys = [
    "county",
    "is_business",
    "product_type",
    "data_block_id"
]

print("\nCLIENT DATASET MERGING KEY ANALYSIS")
print("=" * 55)

for column in merge_keys:

    print(f"\nColumn: {column}")

    print("Unique values:", client_df[column].nunique())

    print("Missing values:", client_df[column].isnull().sum())

print("\nDuplicate combinations of merging keys:")

print(
    client_df.duplicated(subset=merge_keys).sum()
)

print("\nSample merging keys:")
print(client_df[merge_keys].head(10))

# ============================================
# 20. VALIDATE CLIENT AND TRAINING KEY MATCHES
# ============================================

merge_keys = [
    "county",
    "is_business",
    "product_type",
    "data_block_id"
]

train_keys = train_clean[merge_keys].drop_duplicates()

client_keys = client_df[merge_keys].drop_duplicates()

key_check = train_keys.merge(
    client_keys,
    on=merge_keys,
    how="left",
    indicator=True,
    validate="one_to_one"
)

print("\nTRAINING AND CLIENT KEY MATCH CHECK")
print("=" * 55)

print("\nMatching key combinations:")
print((key_check["_merge"] == "both").sum())

print("\nUnmatched training key combinations:")
print((key_check["_merge"] == "left_only").sum())

print("\nTotal distinct training key combinations:")
print(len(train_keys))

print("\nMatch percentage:")
print(
    round(
        (key_check["_merge"] == "both").mean() * 100,
        2
    ),
    "%"
)

# ============================================
# 21. INVESTIGATE UNMATCHED CLIENT KEYS
# ============================================

unmatched_keys = key_check[
    key_check["_merge"] == "left_only"
].drop(columns=["_merge"])

print("\nUNMATCHED TRAINING KEY COMBINATIONS")
print("=" * 55)

print("Total unmatched combinations:", len(unmatched_keys))

print("\nUnmatched records:")
print(unmatched_keys.head(20))

print("\nUnmatched combinations by county:")
print(unmatched_keys["county"].value_counts())

print("\nUnmatched combinations by business type:")
print(unmatched_keys["is_business"].value_counts())

print("\nUnmatched combinations by product type:")
print(unmatched_keys["product_type"].value_counts())

print("\nUnmatched combinations by data block:")
print(unmatched_keys["data_block_id"].value_counts().head(20))

# ============================================
# 22. COMPARE DATA BLOCK RANGES
# ============================================

print("\nDATA BLOCK RANGE COMPARISON")
print("=" * 55)

print("\nTraining dataset:")
print("Minimum data_block_id:", train_clean["data_block_id"].min())
print("Maximum data_block_id:", train_clean["data_block_id"].max())
print("Unique data blocks:", train_clean["data_block_id"].nunique())

print("\nClient dataset:")
print("Minimum data_block_id:", client_df["data_block_id"].min())
print("Maximum data_block_id:", client_df["data_block_id"].max())
print("Unique data blocks:", client_df["data_block_id"].nunique())

print("\nFirst 10 training data blocks:")
print(sorted(train_clean["data_block_id"].unique())[:10])

print("\nFirst 10 client data blocks:")
print(sorted(client_df["data_block_id"].unique())[:10])

# ============================================
# 23. INVESTIGATE REMAINING UNMATCHED KEYS
# ============================================

remaining_unmatched = unmatched_keys[
    unmatched_keys["data_block_id"] >= 2
].copy()

print("\nREMAINING UNMATCHED COMBINATIONS")
print("=" * 55)

print("Total remaining:", len(remaining_unmatched))

print("\nRemaining unmatched records:")
print(
    remaining_unmatched.sort_values(
        ["data_block_id", "county", "is_business", "product_type"]
    ).to_string(index=False)
)

print("\nCount by data block:")
print(
    remaining_unmatched["data_block_id"]
    .value_counts()
    .sort_index()
)

# ============================================
# 24. CHECK CLIENT KEY AVAILABILITY
# ============================================

client_base_keys = client_df[
    ["county", "is_business", "product_type"]
].drop_duplicates()

unmatched_check = unmatched_keys.merge(
    client_base_keys,
    on=["county", "is_business", "product_type"],
    how="left",
    indicator=True
)

print("\nCLIENT KEY AVAILABILITY CHECK")
print("=" * 55)

print(
    "Combinations found in another client data block:",
    (unmatched_check["_merge"] == "both").sum()
)

print(
    "Combinations never found in client data:",
    (unmatched_check["_merge"] == "left_only").sum()
)


# ============================================
# 25. CHECK CLIENT DATA BLOCK COVERAGE
# ============================================

client_block_ranges = (
    client_df.groupby(
        ["county", "is_business", "product_type"]
    )["data_block_id"]
    .agg(["min", "max", "nunique"])
    .reset_index()
)

coverage_check = unmatched_keys.merge(
    client_block_ranges,
    on=["county", "is_business", "product_type"],
    how="left"
)

coverage_check["block_within_range"] = (
    (coverage_check["data_block_id"] >= coverage_check["min"]) &
    (coverage_check["data_block_id"] <= coverage_check["max"])
)

print("\nCLIENT DATA BLOCK COVERAGE")
print("=" * 55)

print(
    "Unmatched blocks within client block range:",
    coverage_check["block_within_range"].sum()
)

print(
    "Unmatched blocks outside client block range:",
    (~coverage_check["block_within_range"]).sum()
)

print("\nSample coverage details:")

print(
    coverage_check[
        [
            "county",
            "is_business",
            "product_type",
            "data_block_id",
            "min",
            "max",
            "nunique",
            "block_within_range"
        ]
    ].head(20).to_string(index=False)
)

# ============================================
# 26. INSPECT UNMATCHED BLOCKS WITHIN RANGE
# ============================================

within_range_unmatched = coverage_check[
    coverage_check["block_within_range"]
].copy()

print("\nUNMATCHED BLOCKS WITHIN CLIENT RANGE")
print("=" * 55)

print("Total:", len(within_range_unmatched))

print("\nDetails:")
print(
    within_range_unmatched[
        [
            "county",
            "is_business",
            "product_type",
            "data_block_id",
            "min",
            "max"
        ]
    ].to_string(index=False)
)

print("\nCount by data block:")
print(
    within_range_unmatched["data_block_id"]
    .value_counts()
    .sort_index()
)

# ============================================
# 27. MERGE CLIENT DATA WITH TRAINING DATA
# ============================================

client_features = client_df[
    merge_keys + ["eic_count", "installed_capacity"]
].copy()

train_with_client = train_clean.merge(
    client_features,
    on=merge_keys,
    how="left",
    validate="many_to_one",
    indicator=True
)

print("\nCLIENT DATA MERGE RESULTS")
print("=" * 55)

print("Original training rows:", len(train_clean))
print("Rows after merge:", len(train_with_client))

print("\nMerge status:")
print(train_with_client["_merge"].value_counts())

print("\nMissing client feature values:")
print(
    train_with_client[
        ["eic_count", "installed_capacity"]
    ].isnull().sum()
)

print("\nDuplicate rows after merge:")
print(train_with_client.duplicated().sum())

print("\nMerged dataset shape:", train_with_client.shape)

# ============================================
# 28. ELECTRICITY PRICE DATASET INSPECTION
# ============================================

electricity_df = pd.read_csv(
    DATA_DIR / "electricity_prices.csv"
)

print("\nELECTRICITY PRICE DATASET")
print("=" * 55)

print("Dataset shape:", electricity_df.shape)

print("\nFirst 5 rows:")
print(electricity_df.head())

print("\nColumn names:")
print(electricity_df.columns.tolist())

print("\nData types:")
print(electricity_df.dtypes)

print("\nMissing values:")
print(electricity_df.isnull().sum())

print("\nDuplicate rows:")
print(electricity_df.duplicated().sum())

print("\nDataset information:")
electricity_df.info()

# ============================================
# 29. CONVERT ELECTRICITY PRICE DATE COLUMNS
# ============================================

date_columns = ["forecast_date", "origin_date"]

for column in date_columns:
    electricity_df[column] = pd.to_datetime(
        electricity_df[column],
        errors="coerce"
    )

print("\nELECTRICITY DATE CONVERSION")
print("=" * 55)

for column in date_columns:
    print(f"\n{column}")
    print("Data type:", electricity_df[column].dtype)
    print("Invalid dates:", electricity_df[column].isnull().sum())
    print("Minimum:", electricity_df[column].min())
    print("Maximum:", electricity_df[column].max())

print("\nElectricity price statistics:")
print(electricity_df["euros_per_mwh"].describe())

# ============================================
# 30. CHECK ELECTRICITY PRICE MERGING KEYS
# ============================================

print("\nELECTRICITY PRICE KEY ANALYSIS")
print("=" * 55)

print("\nUnique forecast dates:")
print(electricity_df["forecast_date"].nunique())

print("\nDuplicate forecast dates:")
print(electricity_df["forecast_date"].duplicated().sum())

print("\nUnique data blocks:")
print(electricity_df["data_block_id"].nunique())

print("\nDuplicate combinations of forecast date and data block:")
print(
    electricity_df.duplicated(
        subset=["forecast_date", "data_block_id"]
    ).sum()
)

print("\nRecords per data block:")
print(
    electricity_df.groupby("data_block_id")
    .size()
    .describe()
)

print("\nFirst 10 electricity records:")
print(electricity_df.head(10))

# ============================================
# 31. VALIDATE ELECTRICITY PRICE MATCHES
# ============================================

train_electricity_keys = train_with_client[
    ["datetime", "data_block_id"]
].drop_duplicates()

electricity_keys = electricity_df[
    ["forecast_date", "data_block_id"]
].drop_duplicates()

electricity_key_check = train_electricity_keys.merge(
    electricity_keys,
    left_on=["datetime", "data_block_id"],
    right_on=["forecast_date", "data_block_id"],
    how="left",
    indicator=True,
    validate="one_to_one"
)

print("\nTRAINING AND ELECTRICITY PRICE MATCH CHECK")
print("=" * 55)

print("\nMatching combinations:")
print(
    (electricity_key_check["_merge"] == "both").sum()
)

print("\nUnmatched combinations:")
print(
    (electricity_key_check["_merge"] == "left_only").sum()
)

print("\nTotal distinct training combinations:")
print(len(train_electricity_keys))

print("\nMatch percentage:")
print(
    round(
        (electricity_key_check["_merge"] == "both").mean() * 100,
        2
    ),
    "%"
)

# ============================================
# 32. TEST ELECTRICITY MATCH USING DATETIME
# ============================================

train_dates = train_with_client[
    ["datetime"]
].drop_duplicates()

electricity_dates = electricity_df[
    ["forecast_date"]
].drop_duplicates()

date_check = train_dates.merge(
    electricity_dates,
    left_on="datetime",
    right_on="forecast_date",
    how="left",
    indicator=True,
    validate="one_to_one"
)

print("\nELECTRICITY DATETIME-ONLY MATCH CHECK")
print("=" * 55)

print(
    "Matching timestamps:",
    (date_check["_merge"] == "both").sum()
)

print(
    "Unmatched timestamps:",
    (date_check["_merge"] == "left_only").sum()
)

print("Total training timestamps:", len(train_dates))

print(
    "Match percentage:",
    round(
        (date_check["_merge"] == "both").mean() * 100,
        2
    ),
    "%"
)

# ============================================
# 33. INSPECT UNMATCHED ELECTRICITY TIMESTAMPS
# ============================================

unmatched_dates = date_check[
    date_check["_merge"] == "left_only"
]["datetime"]

print("\nUNMATCHED ELECTRICITY TIMESTAMPS")
print("=" * 55)

print("Total unmatched timestamps:", len(unmatched_dates))

print("\nUnmatched dates:")
print(unmatched_dates.to_string(index=False))

print("\nEarliest unmatched date:", unmatched_dates.min())
print("Latest unmatched date:", unmatched_dates.max())

print("\nElectricity forecast date range:")
print(electricity_df["forecast_date"].min())
print(electricity_df["forecast_date"].max())

# ============================================
# 34. GROUP UNMATCHED TIMESTAMPS BY DATE
# ============================================

unmatched_dates_df = pd.DataFrame({
    "datetime": unmatched_dates
})

unmatched_dates_df["date"] = (
    unmatched_dates_df["datetime"].dt.date
)

print("\nUNMATCHED TIMESTAMPS BY DATE")
print("=" * 55)

print(
    unmatched_dates_df.groupby("date")
    .size()
    .to_string()
)

print("\nTotal unmatched timestamps:")
print(len(unmatched_dates_df))

# ============================================
# 35. MERGE ELECTRICITY PRICES
# ============================================

electricity_features = electricity_df[
    ["forecast_date", "euros_per_mwh"]
].rename(
    columns={"forecast_date": "datetime"}
)

print("\nDuplicate electricity timestamps:")
print(electricity_features["datetime"].duplicated().sum())

train_with_electricity = train_with_client.merge(
    electricity_features,
    on="datetime",
    how="left",
    validate="many_to_one",
    indicator="electricity_merge"
)

print("\nELECTRICITY MERGE RESULTS")
print("=" * 55)

print("Original training rows:", len(train_with_client))
print("Rows after electricity merge:", len(train_with_electricity))

print("\nMerge status:")
print(train_with_electricity["electricity_merge"].value_counts())

print("\nMissing electricity prices:")
print(train_with_electricity["euros_per_mwh"].isnull().sum())

print("\nDuplicate rows after merge:")
print(train_with_electricity.duplicated().sum())

print("\nFinal merged dataset shape:")
print(train_with_electricity.shape)

# ============================================
# 36. GAS PRICE DATASET INSPECTION
# ============================================

gas_df = pd.read_csv(
    DATA_DIR / "gas_prices.csv"
)

print("\nGAS PRICE DATASET")
print("=" * 55)

print("Dataset shape:", gas_df.shape)

print("\nFirst 5 rows:")
print(gas_df.head())

print("\nColumn names:")
print(gas_df.columns.tolist())

print("\nData types:")
print(gas_df.dtypes)

print("\nMissing values:")
print(gas_df.isnull().sum())

print("\nDuplicate rows:")
print(gas_df.duplicated().sum())

print("\nDataset information:")
gas_df.info()

# ============================================
# 37. CONVERT GAS PRICE DATE COLUMNS
# ============================================

date_columns = ["forecast_date", "origin_date"]

for column in date_columns:
    gas_df[column] = pd.to_datetime(
        gas_df[column],
        errors="coerce"
    )

print("\nGAS DATE CONVERSION")
print("=" * 55)

for column in date_columns:
    print(f"\n{column}")
    print("Data type:", gas_df[column].dtype)
    print("Invalid dates:", gas_df[column].isnull().sum())
    print("Minimum:", gas_df[column].min())
    print("Maximum:", gas_df[column].max())

print("\nLowest gas price statistics:")
print(gas_df["lowest_price_per_mwh"].describe())

print("\nHighest gas price statistics:")
print(gas_df["highest_price_per_mwh"].describe())


# ============================================
# 38. CHECK GAS PRICE MERGING KEYS
# ============================================

print("\nGAS PRICE KEY ANALYSIS")
print("=" * 55)

print("\nUnique forecast dates:")
print(gas_df["forecast_date"].nunique())

print("\nDuplicate forecast dates:")
print(gas_df["forecast_date"].duplicated().sum())

print("\nUnique data blocks:")
print(gas_df["data_block_id"].nunique())

print("\nDuplicate data block IDs:")
print(gas_df["data_block_id"].duplicated().sum())

print("\nFirst 10 gas price records:")
print(gas_df.head(10))

# ============================================
# 39. TRAINING AND GAS PRICE MATCH CHECK
# ============================================

train_blocks = set(train_clean["data_block_id"].unique())
gas_blocks = set(gas_df["data_block_id"].unique())

matching_blocks = train_blocks.intersection(gas_blocks)
unmatched_blocks = train_blocks.difference(gas_blocks)

print("\nTRAINING AND GAS PRICE MATCH CHECK")
print("=" * 55)

print("Matching data blocks:", len(matching_blocks))
print("Unmatched training data blocks:", len(unmatched_blocks))
print("Total training data blocks:", len(train_blocks))

match_percentage = len(matching_blocks) / len(train_blocks) * 100
print("Match percentage:", round(match_percentage, 2), "%")

print("\nUnmatched block IDs:")
print(sorted(unmatched_blocks))
print("\nUNMATCHED TRAINING DATA BLOCK DETAILS")
print("=" * 55)

unmatched_train = train_clean[
    train_clean["data_block_id"].isin(unmatched_blocks)
]

print("Number of training rows:", len(unmatched_train))
print("\nSample unmatched rows:")
print(unmatched_train.head(10))

print("\nUnmatched block row count:")
print(unmatched_train["data_block_id"].value_counts())
print("\nUNMATCHED BLOCK DATE SUMMARY")
print("=" * 55)

print(
    unmatched_train.groupby("data_block_id")["datetime"]
    .agg(["min", "max", "nunique"])
)

print("\nRows by date:")
print(unmatched_train["datetime"].dt.date.value_counts().sort_index())
print("\nGAS PRICE AVAILABILITY FOR BLOCK 0")
print("=" * 55)

print(
    gas_df[gas_df["data_block_id"] == 0]
)
print("\nELECTRICITY PRICE AVAILABILITY FOR BLOCK 0")
print("=" * 55)

print(
    electricity_df[electricity_df["data_block_id"] == 0]
)

# ============================================================
# GAS PRICE MERGE
# ============================================================

print("\nGAS PRICE MERGE")
print("=" * 55)

# Merge gas prices with the training dataset
train_with_gas = train_with_electricity.merge(
    gas_df[
        [
            "data_block_id",
            "lowest_price_per_mwh",
            "highest_price_per_mwh"
        ]
    ],
    on="data_block_id",
    how="left",
    validate="many_to_one",
    indicator="gas_merge"
)

# Check merge results
print("\nOriginal training rows:", len(train_with_electricity))
print("Rows after gas merge:", len(train_with_gas))

print("\nGas merge status:")
print(train_with_gas["gas_merge"].value_counts())

print("\nMissing gas price values:")
print(
    train_with_gas[
        ["lowest_price_per_mwh", "highest_price_per_mwh"]
    ].isnull().sum()
)

print("\nDuplicate rows after gas merge:")
print(train_with_gas.duplicated().sum())

# Final dataset checks
print("\nFINAL MERGED DATASET")
print("=" * 55)

print("Dataset shape:", train_with_gas.shape)

print("\nMissing value counts:")
print(train_with_gas.isnull().sum())

print("\nMissing value percentages:")
print((train_with_gas.isnull().mean() * 100).round(2))

print("\nFirst 5 rows:")
print(train_with_gas.head())

# Save the merged dataset
train_with_gas.to_csv(
    DATA_DIR / "processed_training_data.csv",
    index=False
)

print("\nProcessed dataset saved successfully.")
