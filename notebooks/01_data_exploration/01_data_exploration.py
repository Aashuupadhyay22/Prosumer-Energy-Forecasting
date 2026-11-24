"""
Prosumer Energy Forecasting
01 - Data Exploration and Initial Data Quality Checks

Run this script from the project root, or directly from the notebooks folder.
Expected project structure:
    Prosumer_Energy_Forecasting/
        Data/
        notebooks/
            01_data_exploration.py

This script explores the supplied datasets. It does not train a model,
merge datasets, or remove outliers.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

try:
    # This works when the script is run as a .py file.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback for interactive environments.
    PROJECT_ROOT = Path.cwd()

# If the current folder is already the project root, use it.
if not (PROJECT_ROOT / "Data").exists() and (Path.cwd() / "Data").exists():
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "Data"
PLOT_STYLE = "whitegrid"

sns.set_theme(style=PLOT_STYLE)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 140)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def load_csv(filename, required=False):
    """Load a CSV file from Data/. Missing optional files are skipped."""
    file_path = DATA_DIR / filename

    if not file_path.exists():
        message = f"File not found: {file_path}"
        if required:
            raise FileNotFoundError(message)
        print(f"\nWARNING: {message}")
        return None

    print(f"\nLoading: {file_path.name}")
    return pd.read_csv(file_path)


def explore_dataset(dataframe, dataset_name):
    """Print a consistent overview and data-quality summary."""
    if dataframe is None:
        return

    print(f"\n{'=' * 65}")
    print(f"{dataset_name.upper()} - DATASET EXPLORATION")
    print(f"{'=' * 65}")

    print("\nFirst five rows:")
    print(dataframe.head())

    print("\nDataset shape:")
    print(dataframe.shape)

    print("\nColumn names:")
    print(dataframe.columns.tolist())

    print("\nData types:")
    dataframe.info()

    print("\nMissing values by column:")
    print(dataframe.isnull().sum())

    print("\nTotal missing values:")
    print(dataframe.isnull().sum().sum())

    print("\nDuplicate rows:")
    print(dataframe.duplicated().sum())

    print("\nStatistical summary:")
    print(dataframe.describe(include="all").transpose())


def show_date_summary(dataframe, column, label):
    """Convert a date column and display its date range."""
    if dataframe is None or column not in dataframe.columns:
        return

    dataframe[column] = pd.to_datetime(dataframe[column], errors="coerce")

    print(f"\n{label} date analysis:")
    print("Earliest:", dataframe[column].min())
    print("Latest:", dataframe[column].max())
    print("Unique values:", dataframe[column].nunique())
    print("Unparseable/missing dates:", dataframe[column].isna().sum())


def show_countplot_or_bar(series, title, xlabel, ylabel, figsize=(10, 5)):
    """Plot a grouped mean series as a bar chart."""
    if series is None or series.empty:
        print(f"Skipping empty plot: {title}")
        return

    plt.figure(figsize=figsize)
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 3. TRAINING DATASET: LOAD AND BASIC EXPLORATION
# ============================================================

df = load_csv("train.csv", required=True)

print("\n========== TRAINING DATASET ==========")
print("\nDataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe().transpose())

# Convert datetime once, before all time-based analysis.
if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    print("\nTraining data date range:")
    print("Earliest:", df["datetime"].min())
    print("Latest:", df["datetime"].max())
    print("Unparseable/missing datetimes:", df["datetime"].isna().sum())


# ============================================================
# 4. TARGET VARIABLE: DISTRIBUTION AND ZERO/NEGATIVE VALUES
# ============================================================

if "target" not in df.columns:
    raise KeyError("The training dataset does not contain a 'target' column.")

target_data = df["target"].dropna()

print("\n========== TARGET STATISTICS ==========")
print(target_data.describe())

zero_count = (target_data == 0).sum()
negative_count = (target_data < 0).sum()
positive_count = (target_data > 0).sum()
total_target_count = len(target_data)

print("\n========== ZERO AND NEGATIVE TARGET ANALYSIS ==========")
print("Non-missing target observations:", total_target_count)
print("Zero target values:", zero_count)
print("Negative target values:", negative_count)
print("Positive target values:", positive_count)

if total_target_count:
    print("Percentage of zero values:", round(zero_count / total_target_count * 100, 2), "%")
    print("Percentage of negative values:", round(negative_count / total_target_count * 100, 2), "%")

if "is_consumption" in df.columns:
    print("\nZero target percentage by consumption type:")
    print(
        df.groupby("is_consumption")["target"]
        .apply(lambda values: (values == 0).mean() * 100)
    )

    print("\nTarget summary by consumption type:")
    print(
        df.groupby("is_consumption")["target"]
        .agg(["count", "mean", "median", "min", "max"])
    )

if "is_business" in df.columns:
    print("\nZero target percentage by business type:")
    print(
        df.groupby("is_business")["target"]
        .apply(lambda values: (values == 0).mean() * 100)
    )

    print("\nTarget summary by business type:")
    print(
        df.groupby("is_business")["target"]
        .agg(["count", "mean", "median", "min", "max"])
    )


# Target histogram.
plt.figure(figsize=(10, 5))
sns.histplot(target_data, bins=100, kde=True)
plt.title("Distribution of Energy Target")
plt.xlabel("Energy Target")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Log transform is only valid for values greater than -1.
log_eligible_target = target_data[target_data > -1]
if not log_eligible_target.empty:
    plt.figure(figsize=(10, 5))
    sns.histplot(np.log1p(log_eligible_target), bins=100, kde=True)
    plt.title("Log-Transformed Energy Target")
    plt.xlabel("log(1 + Energy Target)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
else:
    print("\nLog-transformed target plot skipped: no target values greater than -1.")

# Target boxplot.
plt.figure(figsize=(10, 4))
sns.boxplot(x=target_data, showfliers=True)
plt.title("Energy Target - Outlier Overview")
plt.xlabel("Energy Target")
plt.tight_layout()
plt.show()


# ============================================================
# 5. OUTLIER ANALYSIS (IDENTIFICATION ONLY - NO ROWS REMOVED)
# ============================================================

print("\n========== IQR-BASED OUTLIER ANALYSIS ==========")

if not target_data.empty:
    q1 = target_data.quantile(0.25)
    q3 = target_data.quantile(0.75)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outliers = target_data[
        (target_data < lower_limit) | (target_data > upper_limit)
    ]

    print("Q1:", q1)
    print("Q3:", q3)
    print("IQR:", iqr)
    print("Lower limit:", lower_limit)
    print("Upper limit:", upper_limit)
    print("Total outliers:", len(outliers))
    print("Outlier percentage:", round(len(outliers) / len(target_data) * 100, 2), "%")

    print("\nTen highest target observations:")
    columns_to_show = [
        column for column in
        ["target", "county", "is_business", "product_type", "is_consumption", "datetime"]
        if column in df.columns
    ]
    print(df.nlargest(10, "target")[columns_to_show])


# ============================================================
# 6. FEATURE ENGINEERING: DATE/TIME FEATURES
# ============================================================

print("\n========== FEATURE ENGINEERING ==========")

if "datetime" in df.columns:
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

    print("\nNew date/time features:")
    print(
        df[
            [
                "datetime", "year", "month", "day",
                "hour", "day_of_week", "is_weekend"
            ]
        ].head()
    )
    print("\nUpdated training dataset shape:", df.shape)
else:
    print("Datetime features skipped: 'datetime' column not found.")


# ============================================================
# 7. TIME-BASED ENERGY ANALYSIS
# ============================================================

if "hour" in df.columns:
    hourly_target = df.groupby("hour")["target"].mean()
    show_countplot_or_bar(
        hourly_target,
        "Average Energy Target by Hour",
        "Hour of Day",
        "Average Energy Target",
        figsize=(12, 5),
    )

if "day_of_week" in df.columns:
    weekday_target = df.groupby("day_of_week")["target"].mean()
    show_countplot_or_bar(
        weekday_target,
        "Average Energy Target by Day of Week",
        "Day of Week (0 = Monday, 6 = Sunday)",
        "Average Energy Target",
    )

    # Add readable weekday labels to a separate chart.
    plt.figure(figsize=(10, 5))
    weekday_target.reindex(range(7)).plot(kind="bar")
    plt.title("Average Energy Target by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Average Energy Target")
    plt.xticks(
        ticks=range(7),
        labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        rotation=0,
    )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

if "month" in df.columns:
    monthly_target = df.groupby("month")["target"].mean()
    plt.figure(figsize=(10, 5))
    monthly_target.reindex(range(1, 13)).plot(kind="bar")
    plt.title("Average Energy Target by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Energy Target")
    plt.xticks(
        ticks=range(12),
        labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        rotation=0,
    )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

if "year" in df.columns:
    yearly_target = df.groupby("year")["target"].mean()
    show_countplot_or_bar(
        yearly_target,
        "Average Energy Target by Year",
        "Year",
        "Average Energy Target",
        figsize=(8, 5),
    )
    print("\nYearly average energy target:")
    print(yearly_target)

if "datetime" in df.columns:
    daily_target = df.groupby(df["datetime"].dt.floor("D"))["target"].mean()
    plt.figure(figsize=(14, 5))
    daily_target.plot()
    plt.title("Average Daily Energy Target")
    plt.xlabel("Date")
    plt.ylabel("Average Energy Target")
    plt.tight_layout()
    plt.show()


# ============================================================
# 8. BUSINESS, CONSUMPTION, AND PRODUCT-TYPE ANALYSIS
# ============================================================

if "is_business" in df.columns:
    business_average = df.groupby("is_business")["target"].mean()
    show_countplot_or_bar(
        business_average,
        "Average Energy Target: Business vs Non-Business",
        "Is Business (0 = No, 1 = Yes)",
        "Average Energy Target",
        figsize=(8, 5),
    )

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="is_business", y="target", showfliers=False)
    plt.title("Energy Target by Business Type")
    plt.xlabel("0 = Non-Business, 1 = Business")
    plt.ylabel("Energy Target")
    plt.tight_layout()
    plt.show()

if "is_consumption" in df.columns:
    consumption_average = df.groupby("is_consumption")["target"].mean()
    show_countplot_or_bar(
        consumption_average,
        "Average Energy Target: Consumption vs Production",
        "Is Consumption (0 = Production, 1 = Consumption)",
        "Average Energy Target",
        figsize=(8, 5),
    )

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="is_consumption", y="target", showfliers=False)
    plt.title("Energy Target by Consumption Type")
    plt.xlabel("0 = Production, 1 = Consumption")
    plt.ylabel("Energy Target")
    plt.tight_layout()
    plt.show()

if "product_type" in df.columns:
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x="product_type", y="target", showfliers=False)
    plt.title("Energy Target by Product Type")
    plt.xlabel("Product Type")
    plt.ylabel("Energy Target")
    plt.tight_layout()
    plt.show()


# ============================================================
# 9. CORRELATION ANALYSIS
# ============================================================

print("\n========== CORRELATION ANALYSIS ==========")

numeric_df = df.select_dtypes(include=[np.number]).copy()

# Identifier columns are not meaningful continuous measurements.
identifier_columns = [
    column for column in numeric_df.columns
    if column.lower() == "id" or column.lower().endswith("_id")
]
numeric_df = numeric_df.drop(columns=identifier_columns, errors="ignore")

if "target" in numeric_df.columns and numeric_df.shape[1] >= 2:
    correlation_matrix = numeric_df.corr()

    print("\nCorrelation with target:")
    print(correlation_matrix["target"].sort_values(ascending=False))

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title("Numerical Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()
else:
    print("Correlation analysis skipped: not enough numeric columns.")


# ============================================================
# 10. DATA QUALITY REVIEW AND CLEANED TARGET DATASET
# ============================================================

print("\n========== FINAL TRAINING DATA QUALITY REVIEW ==========")

print("\nMissing values by column:")
print(df.isnull().sum())

missing_by_column = df.isnull().sum()
missing_by_column = missing_by_column[missing_by_column > 0]

print("\nColumns containing missing values:")
print(missing_by_column)

if len(df):
    print("\nMissing value percentage:")
    print((missing_by_column / len(df) * 100).round(2))

print("\nDuplicate rows:", df.duplicated().sum())

# Remove only rows whose target is missing.
# Other missing values are reported, not automatically imputed or removed.
df_clean = df.dropna(subset=["target"]).copy()

print("\nOriginal dataset shape:", df.shape)
print("Dataset shape after removing missing targets:", df_clean.shape)
print("Remaining missing targets:", df_clean["target"].isna().sum())

print("\nMissing values in cleaned-target dataset:")
print(df_clean.isnull().sum().sort_values(ascending=False))

print("\nTotal remaining missing values:", df_clean.isnull().sum().sum())

print("\nNegative target values in cleaned-target dataset:")
print((df_clean["target"] < 0).sum())

for column in ["is_consumption", "is_business", "product_type", "county"]:
    if column in df_clean.columns:
        print(f"\nUnique values in {column}:")
        print(sorted(df_clean[column].dropna().unique()))


# ============================================================
# 11. CLIENT DATASET
# ============================================================

client_df = load_csv("client.csv")
if client_df is not None:
    explore_dataset(client_df, "Client")

    show_date_summary(client_df, "date", "Client")

    client_merge_keys = [
        column for column in
        ["date", "county", "is_business", "product_type"]
        if column in client_df.columns
    ]
    if client_merge_keys:
        print("\nDuplicate client merge-key rows:")
        print(client_df.duplicated(subset=client_merge_keys).sum())


# ============================================================
# 12. ELECTRICITY PRICE DATASET
# ============================================================

electricity_df = load_csv("electricity_prices.csv")
if electricity_df is not None:
    explore_dataset(electricity_df, "Electricity prices")
    show_date_summary(electricity_df, "forecast_date", "Electricity forecast")
    show_date_summary(electricity_df, "origin_date", "Electricity origin")

    if "data_block_id" in electricity_df.columns:
        print(
            "\nDuplicate electricity data_block_id values:",
            electricity_df["data_block_id"].duplicated().sum(),
        )


# ============================================================
# 13. GAS PRICE DATASET
# ============================================================

gas_df = load_csv("gas_prices.csv")
if gas_df is not None:
    explore_dataset(gas_df, "Gas prices")
    show_date_summary(gas_df, "forecast_date", "Gas forecast")
    show_date_summary(gas_df, "origin_date", "Gas origin")

    if "data_block_id" in gas_df.columns:
        print(
            "\nDuplicate gas data_block_id values:",
            gas_df["data_block_id"].duplicated().sum(),
        )


# ============================================================
# 14. HISTORICAL WEATHER DATASET
# ============================================================

historical_weather_df = load_csv("historical_weather.csv")
if historical_weather_df is not None:
    explore_dataset(historical_weather_df, "Historical weather")
    show_date_summary(historical_weather_df, "datetime", "Historical weather")

    coordinate_columns = [
        column for column in ["latitude", "longitude"]
        if column in historical_weather_df.columns
    ]
    if len(coordinate_columns) == 2:
        print(
            "\nNumber of unique weather locations:",
            historical_weather_df[coordinate_columns].drop_duplicates().shape[0],
        )

    weather_key = [
        column for column in ["datetime", "latitude", "longitude"]
        if column in historical_weather_df.columns
    ]

    if len(weather_key) == 3:
        duplicate_weather = historical_weather_df[
            historical_weather_df.duplicated(subset=weather_key, keep=False)
        ].sort_values(by=weather_key)

        print("\nNumber of repeated weather records:", len(duplicate_weather))
        if not duplicate_weather.empty:
            print("\nSample repeated weather records:")
            print(duplicate_weather.head(20).to_string(index=False))


# ============================================================
# 15. FORECAST WEATHER DATASET
# ============================================================

forecast_weather_df = load_csv("forecast_weather.csv")
if forecast_weather_df is not None:
    explore_dataset(forecast_weather_df, "Forecast weather")
    show_date_summary(forecast_weather_df, "origin_datetime", "Forecast weather origin")
    show_date_summary(forecast_weather_df, "forecast_datetime", "Forecast weather target")

    coordinate_columns = [
        column for column in ["latitude", "longitude"]
        if column in forecast_weather_df.columns
    ]
    if len(coordinate_columns) == 2:
        print(
            "\nNumber of unique forecast weather locations:",
            forecast_weather_df[coordinate_columns].drop_duplicates().shape[0],
        )

    if "hours_ahead" in forecast_weather_df.columns:
        print("\nForecast hours-ahead values:")
        print(sorted(forecast_weather_df["hours_ahead"].dropna().unique()))

        print("\nForecast hours-ahead counts:")
        print(forecast_weather_df["hours_ahead"].value_counts().sort_index())

        print("\nForecast horizon statistics:")
        print(forecast_weather_df["hours_ahead"].describe())

        plt.figure(figsize=(12, 5))
        sns.countplot(data=forecast_weather_df, x="hours_ahead", color="skyblue")
        plt.title("Distribution of Forecast Hours Ahead")
        plt.xlabel("Hours Ahead")
        plt.ylabel("Number of Records")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# ============================================================
# 16. WEATHER STATION TO COUNTY MAPPING
# ============================================================

weather_mapping_df = load_csv("weather_station_to_county_mapping.csv")
if weather_mapping_df is not None:
    explore_dataset(weather_mapping_df, "Weather station to county mapping")

    if "county" in weather_mapping_df.columns:
        missing_county = weather_mapping_df[weather_mapping_df["county"].isna()]
        print("\nRows with missing county mapping:", len(missing_county))

        coordinate_columns = [
            column for column in ["latitude", "longitude"]
            if column in missing_county.columns
        ]
        if coordinate_columns:
            print("\nCoordinates with missing county mapping:")
            print(missing_county[coordinate_columns].head(20))


# ============================================================
# 17. COUNTY ID TO NAME MAPPING (JSON)
# ============================================================

county_json_path = DATA_DIR / "county_id_to_name_map.json"

if county_json_path.exists():
    with open(county_json_path, "r", encoding="utf-8") as file:
        county_mapping = json.load(file)

    county_mapping_df = pd.DataFrame(
        list(county_mapping.items()),
        columns=["county_id", "county_name"],
    )

    explore_dataset(county_mapping_df, "County ID to name mapping")
else:
    print(f"\nWARNING: County mapping JSON not found: {county_json_path}")


# ============================================================
# 18. REVEALED TARGETS DATASET
# ============================================================

revealed_targets_df = load_csv("example_test_files/revealed_targets.csv")
if revealed_targets_df is not None:
    explore_dataset(revealed_targets_df, "Revealed targets")


# ============================================================
# 19. SAMPLE SUBMISSION DATASET
# ============================================================

sample_submission_df = load_csv("example_test_files/sample_submission.csv")
if sample_submission_df is not None:
    explore_dataset(sample_submission_df, "Sample submission")


# ============================================================
# 20. TEST DATASET
# ============================================================

test_df = load_csv("example_test_files/test.csv")
if test_df is not None:
    explore_dataset(test_df, "Test dataset")


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("DATA EXPLORATION COMPLETED")
print("=" * 65)
print("Training dataset shape:", df.shape)
print("Training dataset shape after dropping missing targets:", df_clean.shape)
print("No outliers or other missing values were automatically removed.")
print("Review the findings before deciding on preprocessing or modelling.")
