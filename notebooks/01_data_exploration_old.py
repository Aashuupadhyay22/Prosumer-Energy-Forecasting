
import pandas as pd
import numpy as np

# Load the training dataset
data_path = "Data/train.csv"

df = pd.read_csv(data_path)

# Display basic information
print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATASET INFORMATION ---")
df.info()

print("\n========== DATA QUALITY ==========")

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Statistical summary
print("\nStatistical summary:")
print(df.describe())

# Target distribution
print("\nTarget statistics:")
print(df["target"].describe())

# Date range
print("\nDate range:")
print(df["datetime"].min())
print(df["datetime"].max())




import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Target distribution
sns.histplot(df["target"], bins=100, kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Distribution of Energy Target")

# 2. Consumption vs Production
sns.boxplot(x="is_consumption", y="target", data=df, ax=axes[0, 1])
axes[0, 1].set_title("Consumption vs Production")

# 3. Business vs Non-business
sns.boxplot(x="is_business", y="target", data=df, ax=axes[1, 0])
axes[1, 0].set_title("Business vs Non-Business")

# 4. Average daily energy target
df["datetime"] = pd.to_datetime(df["datetime"])
daily = df.groupby(df["datetime"].dt.date)["target"].mean()

axes[1, 1].plot(daily.index, daily.values)
axes[1, 1].set_title("Average Daily Energy Target")
axes[1, 1].set_xlabel("Date")
axes[1, 1].set_ylabel("Average Target")

plt.tight_layout()
plt.show()

import numpy as np

plt.figure(figsize=(10, 5))

sns.histplot(np.log1p(df["target"]), bins=100, kde=True)

plt.title("Log-Transformed Energy Target Distribution")
plt.xlabel("log(1 + Energy Target)")
plt.ylabel("Frequency")

plt.show()



# Zero-value analysis

zero_count = (df['target'] == 0).sum()
total_count = len(df)

zero_percentage = (zero_count / total_count) * 100

print("Zero Target Analysis")
print("--------------------")
print("Total observations:", total_count)
print("Zero target values:", zero_count)
print("Non-zero target values:", total_count - zero_count)
print("Percentage of zero values:", round(zero_percentage, 2), "%")

# Zero target analysis by consumption
print("\nZero targets by consumption:")
print(df.groupby("is_consumption")["target"].apply(
    lambda x: (x == 0).mean() * 100
))

# Zero target analysis by business
print("\nZero targets by business:")
print(df.groupby("is_business")["target"].apply(
    lambda x: (x == 0).mean() * 100
))


# ==========================================
# FEATURE ENGINEERING
# ==========================================

# Convert datetime column
df["datetime"] = pd.to_datetime(df["datetime"])

# Extract date and time features
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek

# Weekend indicator (1 = Weekend, 0 = Weekday)
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# Display new features
print("\nNew Features:")
print(df[[
    "datetime",
    "year",
    "month",
    "day",
    "hour",
    "day_of_week",
    "is_weekend"
]].head())

print("\nUpdated Dataset Shape:")
print(df.shape)

# Hourly Energy Analysis

hourly = df.groupby("hour")["target"].mean()

plt.figure(figsize=(12, 5))
hourly.plot(kind="bar")

plt.title("Average Energy Target by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Energy Target")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)

plt.show()

# Average Energy Target by Day of Week

weekly = df.groupby("day_of_week")["target"].mean()

plt.figure(figsize=(10, 5))
weekly.plot(kind="bar")

plt.title("Average Energy Target by Day of Week")
plt.xlabel("Day of Week (0=Monday, 6=Sunday)")
plt.ylabel("Average Energy Target")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)

plt.show()

# Monthly energy analysis

monthly = df.groupby(
    df["datetime"].dt.month
)["target"].mean()

plt.figure(figsize=(10, 5))
monthly.plot(kind="bar")

plt.title("Average Energy Target by Month")
plt.xlabel("Month")
plt.ylabel("Average Energy Target")
plt.xticks(
    ticks=range(12),
    labels=[
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ],
    rotation=0
)

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Yearly energy analysis

yearly = df.groupby("year")["target"].mean()

plt.figure(figsize=(8, 5))
yearly.plot(kind="bar")

plt.title("Average Energy Target by Year")
plt.xlabel("Year")
plt.ylabel("Average Energy Target")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# Average energy target by business category

business_avg = df.groupby("is_business")["target"].mean()

plt.figure(figsize=(8, 5))
business_avg.plot(kind="bar")

plt.title("Average Energy Target: Business vs Non-Business")
plt.xlabel("Is Business (0 = No, 1 = Yes)")
plt.ylabel("Average Energy Target")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Consumption vs Production analysis

consumption_avg = df.groupby("is_consumption")["target"].mean()

plt.figure(figsize=(8, 5))
consumption_avg.plot(kind="bar")

plt.title("Average Energy Target: Consumption vs Production")
plt.xlabel("Is Consumption (0 = Production, 1 = Consumption)")
plt.ylabel("Average Energy Target")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# Correlation analysis

import seaborn as sns
import matplotlib.pyplot as plt

numeric_df = df.select_dtypes(include=["number"])

correlation = numeric_df.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Missing value analysis
print("\nMissing Values:")
print(df.isnull().sum())

# Total missing values
print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# Duplicate analysis
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Dataset information
print("\nDataset Information:")
print(df.info())
# Missing Value and Duplicate Analysis

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTotal Missing Values:", df.isnull().sum().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Detailed Missing Value Analysis

missing = df.isnull().sum()

missing = missing[missing > 0]

print("\nColumns with Missing Values:")
print(missing)

print("\nMissing Value Percentage:")
print((missing / len(df)) * 100)



# Feature Selection Analysis

print("\n========== FEATURE SELECTION ANALYSIS ==========")

# Target variable
target_column = "target"

# Display all columns
print("\nAll Dataset Columns:")
print(df.columns.tolist())

# Identify target and input features
feature_columns = [
    col for col in df.columns
    if col != target_column
]

print("\nTarget Variable:")
print(target_column)

print("\nInput Features:")
print(feature_columns)

# Identify numerical and categorical columns
numerical_features = df[feature_columns].select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = df[feature_columns].select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)

# Identify potential ID columns
id_columns = [
    col for col in df.columns
    if "id" in col.lower()
]

print("\nPotential ID Columns:")
print(id_columns)





# Detailed Feature Inspection

print("\n========== FEATURE INSPECTION ==========")

for col in df.columns:

    print(f"\nFeature: {col}")
    print("Data Type:", df[col].dtype)
    print("Unique Values:", df[col].nunique())
    print("Sample Values:")
    print(df[col].dropna().unique()[:10])

# Target Distribution Analysis

import matplotlib.pyplot as plt
import seaborn as sns

target_data = df["target"].dropna()

print("\n========== TARGET DISTRIBUTION ==========")

print(target_data.describe())

# Histogram
plt.figure(figsize=(10, 5))
sns.histplot(target_data, bins=50, kde=True)

plt.title("Distribution of Energy Target")
plt.xlabel("Energy Target")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x=target_data)

plt.title("Energy Target - Outlier Analysis")
plt.xlabel("Energy Target")

plt.tight_layout()
plt.show()

# IQR-Based Outlier Analysis

print("\n========== OUTLIER ANALYSIS ==========")

Q1 = target_data.quantile(0.25)
Q3 = target_data.quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)

print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

# Identify outliers
outliers = target_data[
    (target_data < lower_limit) |
    (target_data > upper_limit)
]

print("\nTotal Outliers:", len(outliers))
print("Outlier Percentage:", (len(outliers) / len(target_data)) * 100)

print("\nTop 10 Highest Target Values:")
print(df.nlargest(10, "target")[
    ["target", "county", "is_business",
     "product_type", "is_consumption", "datetime"]
])
# Target Distribution Analysis

plt.figure(figsize=(12, 5))

sns.histplot(df["target"].dropna(), bins=100, kde=True)

plt.title("Distribution of Energy Target")
plt.xlabel("Energy Target")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# Log Transformation

plt.figure(figsize=(12, 5))

sns.histplot(np.log1p(df["target"].dropna()), bins=100, kde=True)

plt.title("Log Transformed Energy Target")
plt.xlabel("Log(1 + Energy Target)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# Zero and Negative Target Analysis

print("\nZero Target Values:")
print((df["target"] == 0).sum())

print("\nNegative Target Values:")
print((df["target"] < 0).sum())

print("\nPositive Target Values:")
print((df["target"] > 0).sum())

print("\nPercentage of Zero Values:")
print((df["target"] == 0).mean() * 100)

print("\nTarget Statistics:")
print(df["target"].describe())

# Zero Target Analysis by Consumption Type

print("\nZero Target by Consumption Type:")

print(
    df.groupby("is_consumption")["target"]
    .agg(
        total="count",
        zero_values=lambda x: (x == 0).sum(),
        mean="mean",
        median="median"
    )
)

# Zero Target by Business Type

print("\nZero Target by Business Type:")

print(
    df.groupby("is_business")["target"]
    .agg(
        total="count",
        zero_values=lambda x: (x == 0).sum(),
        mean="mean",
        median="median"
    )
)


# Target Distribution by Consumption Type

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df,
    x="is_consumption",
    y="target",
    showfliers=False
)

plt.title("Energy Target by Consumption Type")
plt.xlabel("0 = Production, 1 = Consumption")
plt.ylabel("Energy Target")

plt.tight_layout()
plt.show()
# Energy Target by Product Type

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df,
    x="product_type",
    y="target",
    showfliers=False
)

plt.title("Energy Target by Product Type")
plt.xlabel("Product Type")
plt.ylabel("Energy Target")

plt.tight_layout()
plt.show()
# Energy Target by Business Type

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="is_business",
    y="target",
    showfliers=False
)

plt.title("Energy Target by Business Type")
plt.xlabel("0 = Non-Business, 1 = Business")
plt.ylabel("Energy Target")

plt.tight_layout()
plt.show()

# Hourly Energy Target Analysis

hourly_target = df.groupby("hour")["target"].mean()

plt.figure(figsize=(12, 6))

sns.lineplot(
    x=hourly_target.index,
    y=hourly_target.values,
    marker="o"
)

plt.title("Average Energy Target by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Energy Target")
plt.xticks(range(24))
plt.grid(True)

plt.tight_layout()
plt.show()


# Average Energy Target by Day of Week

daily_target = df.groupby("day_of_week")["target"].mean()

plt.figure(figsize=(10, 6))

sns.barplot(
    x=daily_target.index,
    y=daily_target.values
)

plt.title("Average Energy Target by Day of Week")
plt.xlabel("Day of Week (0 = Monday)")
plt.ylabel("Average Energy Target")

plt.xticks(range(7),
           ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

plt.tight_layout()
plt.show()

# Monthly Energy Target Analysis

print("\nDatetime Data Type:")
print(df["datetime"].dtype)
# Extract Month and Year

df["month"] = df["datetime"].dt.month
df["year"] = df["datetime"].dt.year

print("\nMonth Distribution:")
print(df["month"].value_counts().sort_index())

print("\nYear Distribution:")
print(df["year"].value_counts().sort_index())


# Monthly Average Energy Target

monthly_target = df.groupby("month")["target"].mean()

plt.figure(figsize=(12, 5))

sns.barplot(
    x=monthly_target.index,
    y=monthly_target.values
)

plt.title("Average Energy Target by Month")
plt.xlabel("Month")
plt.ylabel("Average Energy Target")
plt.xticks(range(12), [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
])

plt.tight_layout()
plt.show()
# Yearly Average Energy Target

yearly_target = df.groupby("year")["target"].mean()

plt.figure(figsize=(9, 5))

sns.barplot(
    x=yearly_target.index,
    y=yearly_target.values
)

plt.title("Average Energy Target by Year")
plt.xlabel("Year")
plt.ylabel("Average Energy Target")

plt.tight_layout()
plt.show()

print("\nYearly Average Energy Target:")
print(yearly_target)
# Yearly Energy Target Analysis

yearly_target = df.groupby("year")["target"].mean()

print("\nAverage Energy Target by Year:")
print(yearly_target)

plt.figure(figsize=(10, 5))

sns.barplot(
    x=yearly_target.index,
    y=yearly_target.values
)

plt.title("Average Energy Target by Year")
plt.xlabel("Year")
plt.ylabel("Average Energy Target")

plt.tight_layout()
plt.show()



# Correlation Analysis

# Select Numerical Columns
numeric_df = df.select_dtypes(include=["number"])

# Calculate Correlation Matrix
correlation_matrix = numeric_df.corr()

# Display Target Correlation
print("\nCorrelation with Energy Target:")
print(
    correlation_matrix["target"]
    .sort_values(ascending=False)
)

# Plot Correlation Heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()



# Missing Target Analysis

missing_target = df["target"].isnull().sum()

missing_percentage = (
    missing_target / len(df)
) * 100

print("\nMissing Target Count:", missing_target)
print("Missing Target Percentage:", missing_percentage)



# Remove Rows with Missing Target

df_clean = df.dropna(subset=["target"]).copy()

print("\nOriginal Dataset Shape:", df.shape)
print("Cleaned Dataset Shape:", df_clean.shape)

print(
    "Remaining Missing Targets:",
    df_clean["target"].isnull().sum()
)
# Remove Rows with Missing Target
df_clean = df.dropna(subset=["target"]).copy()

print("\nOriginal Dataset Shape:", df.shape)
print("Cleaned Dataset Shape:", df_clean.shape)

# Check Missing Values After Cleaning
print("\nMissing Values in Cleaned Dataset:")
print(df_clean.isnull().sum().sort_values(ascending=False))

print("\nTotal Missing Values:", df_clean.isnull().sum().sum())

# Data Consistency Checks

print("\n--- Data Consistency Checks ---")

print("\nNegative Target Values:")
print((df_clean["target"] < 0).sum())

print("\nUnique is_consumption Values:")
print(df_clean["is_consumption"].unique())

print("\nUnique is_business Values:")
print(df_clean["is_business"].unique())

print("\nUnique Product Types:")
print(df_clean["product_type"].unique())

print("\nUnique County Values:")
print(sorted(df_clean["county"].unique()))


# Dataset Column Inspection

print("\n--- Dataset Columns and Data Types ---")

print(df_clean.dtypes)

print("\n--- First Five Rows of Cleaned Dataset ---")

print(df_clean.head())


# ==========================================
# STEP 15: CLIENT DATASET EXPLORATION
# ==========================================

# Load client dataset
client_df = pd.read_csv("Data/client.csv")

# Display first five rows
print("\n--- Client Dataset ---")
print(client_df.head())

# Dataset dimensions
print("\nClient Dataset Shape:")
print(client_df.shape)

# Column names
print("\nClient Dataset Columns:")
print(client_df.columns.tolist())

# Data types
print("\nClient Dataset Information:")
print(client_df.info())

# Missing values
print("\nMissing Values:")
print(client_df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:")
print(client_df.describe())



# Client Dataset Date Analysis

client_df["date"] = pd.to_datetime(client_df["date"])

print("\n--- Client Date Range ---")

print("Earliest Date:", client_df["date"].min())
print("Latest Date:", client_df["date"].max())

print("\nUnique Dates:")
print(client_df["date"].nunique())

print("\nDuplicate Rows:")
print(client_df.duplicated().sum())

print("\nDuplicate Merge Keys:")
print(
    client_df.duplicated(
        subset=[
            "date",
            "county",
            "is_business",
            "product_type"
        ]
    ).sum()
)



# ==========================================
# STEP 17: ELECTRICITY PRICE DATASET
# ==========================================

electricity_df = pd.read_csv(
    "Data/electricity_prices.csv"
)

print("\n--- Electricity Price Dataset ---")

print(electricity_df.head())

print("\nDataset Shape:")
print(electricity_df.shape)

print("\nColumns:")
print(electricity_df.columns.tolist())

print("\nData Types:")
print(electricity_df.dtypes)

print("\nMissing Values:")
print(electricity_df.isnull().sum())

print("\nStatistical Summary:")
print(electricity_df.describe())


# ==========================================
# STEP 18: ELECTRICITY PRICE DATE ANALYSIS
# ==========================================

# Convert date columns to datetime
electricity_df["forecast_date"] = pd.to_datetime(
    electricity_df["forecast_date"]
)

electricity_df["origin_date"] = pd.to_datetime(
    electricity_df["origin_date"]
)

print("\n--- Electricity Price Date Analysis ---")

print("Earliest Forecast Date:",
      electricity_df["forecast_date"].min())

print("Latest Forecast Date:",
      electricity_df["forecast_date"].max())

print("\nUnique Forecast Dates:",
      electricity_df["forecast_date"].nunique())

print("\nUnique Origin Dates:",
      electricity_df["origin_date"].nunique())

print("\nDuplicate Rows:",
      electricity_df.duplicated().sum())

print("\nDuplicate Data Block IDs:",
      electricity_df["data_block_id"].duplicated().sum())

# ==========================================
# STEP 19: GAS PRICE DATASET EXPLORATION
# ==========================================

gas_df = pd.read_csv("Data/gas_prices.csv")

print("\n--- Gas Price Dataset ---")

print(gas_df.head())

print("\nDataset Shape:")
print(gas_df.shape)

print("\nColumns:")
print(gas_df.columns.tolist())

print("\nData Types:")
print(gas_df.dtypes)

print("\nMissing Values:")
print(gas_df.isnull().sum())

print("\nStatistical Summary:")
print(gas_df.describe())

# ==========================================
# STEP 20: GAS PRICE DATE ANALYSIS
# ==========================================

# Convert date columns
gas_df["forecast_date"] = pd.to_datetime(
    gas_df["forecast_date"]
)

gas_df["origin_date"] = pd.to_datetime(
    gas_df["origin_date"]
)

print("\n--- Gas Price Date Analysis ---")

print("Earliest Forecast Date:",
      gas_df["forecast_date"].min())

print("Latest Forecast Date:",
      gas_df["forecast_date"].max())

print("\nUnique Forecast Dates:",
      gas_df["forecast_date"].nunique())

print("\nDuplicate Rows:",
      gas_df.duplicated().sum())

print("\nDuplicate Data Block IDs:",
      gas_df["data_block_id"].duplicated().sum())


# ==========================================
# STEP 21: HISTORICAL WEATHER DATASET
# ==========================================

historical_weather_df = pd.read_csv(
    "Data/historical_weather.csv"
)

print("\n--- Historical Weather Dataset ---")

print(historical_weather_df.head())

print("\nDataset Shape:")
print(historical_weather_df.shape)

print("\nColumns:")
print(historical_weather_df.columns.tolist())

print("\nData Types:")
print(historical_weather_df.dtypes)

print("\nMissing Values:")
print(historical_weather_df.isnull().sum())

print("\nStatistical Summary:")
print(historical_weather_df.describe())



# ==========================================
# STEP 22: HISTORICAL WEATHER DATE ANALYSIS
# ==========================================

# Convert datetime column
historical_weather_df["datetime"] = pd.to_datetime(
    historical_weather_df["datetime"]
)

print("\n--- Historical Weather Date Analysis ---")

print("Earliest Date:",
      historical_weather_df["datetime"].min())

print("Latest Date:",
      historical_weather_df["datetime"].max())

print("\nUnique Datetimes:",
      historical_weather_df["datetime"].nunique())

print("\nUnique Weather Stations:")
print(
    historical_weather_df[
        ["latitude", "longitude"]
    ].drop_duplicates().shape[0]
)

print("\nDuplicate Rows:",
      historical_weather_df.duplicated().sum())

print("\nDuplicate Weather Records:")
print(
    historical_weather_df.duplicated(
        subset=["datetime", "latitude", "longitude"]
    ).sum()
)

# ==========================================
# STEP 23: INVESTIGATE DUPLICATE WEATHER RECORDS
# ==========================================

duplicate_weather = historical_weather_df[
    historical_weather_df.duplicated(
        subset=["datetime", "latitude", "longitude"],
        keep=False
    )
].sort_values(
    by=["datetime", "latitude", "longitude"]
)

print("\n--- Repeated Weather Records ---")

print(duplicate_weather.to_string(index=False))

print("\nNumber of Repeated Records:")
print(len(duplicate_weather))



# Inspect Historical Weather Columns

print("\n--- All Historical Weather Columns ---")

print(historical_weather_df.columns.tolist())

print("\n--- Records at Repeated Coordinates and Time ---")

mask = (
    (historical_weather_df["datetime"] == "2022-11-12 15:00:00")
    & (historical_weather_df["latitude"] == 57.6)
    & (historical_weather_df["longitude"] == 23.2)
)

print(historical_weather_df.loc[mask].to_string(index=False))


# ==========================================
# STEP 24: FORECAST WEATHER DATASET
# ==========================================

forecast_weather_df = pd.read_csv(
    "Data/forecast_weather.csv"
)

print("\n--- Forecast Weather Dataset ---")

print(forecast_weather_df.head())

print("\nDataset Shape:")
print(forecast_weather_df.shape)

print("\nColumns:")
print(forecast_weather_df.columns.tolist())

print("\nData Types:")
print(forecast_weather_df.dtypes)

print("\nMissing Values:")
print(forecast_weather_df.isnull().sum())

print("\nStatistical Summary:")
print(forecast_weather_df.describe())



# ==========================================
# STEP 25: FORECAST WEATHER DATE ANALYSIS
# ==========================================

# Convert date columns
forecast_weather_df["origin_datetime"] = pd.to_datetime(
    forecast_weather_df["origin_datetime"]
)

forecast_weather_df["forecast_datetime"] = pd.to_datetime(
    forecast_weather_df["forecast_datetime"]
)

print("\n--- Forecast Weather Date Analysis ---")

print("Earliest Origin Date:")
print(forecast_weather_df["origin_datetime"].min())

print("\nLatest Origin Date:")
print(forecast_weather_df["origin_datetime"].max())

print("\nEarliest Forecast Date:")
print(forecast_weather_df["forecast_datetime"].min())

print("\nLatest Forecast Date:")
print(forecast_weather_df["forecast_datetime"].max())

print("\nUnique Origin Datetimes:")
print(forecast_weather_df["origin_datetime"].nunique())

print("\nUnique Forecast Datetimes:")
print(forecast_weather_df["forecast_datetime"].nunique())

print("\nUnique Weather Locations:")
print(
    forecast_weather_df[
        ["latitude", "longitude"]
    ].drop_duplicates().shape[0]
)

print("\nHours Ahead Values:")
print(sorted(forecast_weather_df["hours_ahead"].unique()))

# ==========================================
# STEP 26: FORECAST HORIZON ANALYSIS
# ==========================================

print("\n--- Forecast Hours Ahead Analysis ---")

print(
    forecast_weather_df["hours_ahead"]
    .value_counts()
    .sort_index()
)

print("\nForecast Horizon Statistics:")

print(
    forecast_weather_df["hours_ahead"].describe()
)

# Visualize forecast horizon distribution

plt.figure(figsize=(12, 5))

sns.countplot(
    data=forecast_weather_df,
    x="hours_ahead",
    color="skyblue"
)

plt.title("Distribution of Forecast Hours Ahead")
plt.xlabel("Hours Ahead")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ==========================================
# STEP 27: WEATHER STATION COUNTY MAPPING
# ==========================================

weather_mapping_df = pd.read_csv(
    "Data/weather_station_to_county_mapping.csv"
)

print("\n--- Weather Station Mapping ---")

print(weather_mapping_df.head())

print("\nDataset Shape:")
print(weather_mapping_df.shape)

print("\nColumns:")
print(weather_mapping_df.columns.tolist())

print("\nData Types:")
print(weather_mapping_df.dtypes)

print("\nMissing Values:")
print(weather_mapping_df.isnull().sum())

print("\nDuplicate Rows:")
print(weather_mapping_df.duplicated().sum())



# ==========================================
# STEP 27: WEATHER STATION COUNTY MAPPING
# ==========================================

weather_mapping_df = pd.read_csv(
    "Data/weather_station_to_county_mapping.csv"
)

print("\n--- Weather Station Mapping ---")

print(weather_mapping_df.head())

print("\nDataset Shape:")
print(weather_mapping_df.shape)

print("\nColumns:")
print(weather_mapping_df.columns.tolist())

print("\nData Types:")
print(weather_mapping_df.dtypes)

print("\nMissing Values:")
print(weather_mapping_df.isnull().sum())

print("\nDuplicate Rows:")
print(weather_mapping_df.duplicated().sum())



# ==========================================
# STEP 28: INVESTIGATE MISSING COUNTY MAPPING
# ==========================================

missing_county = weather_mapping_df[
    weather_mapping_df["county"].isnull()
]

print("\n--- Records with Missing County ---")

print(missing_county.head(20))

print("\nTotal Missing County Records:")
print(len(missing_county))

print("\nMissing County Coordinates:")
print(
    missing_county[
        ["latitude", "longitude"]
    ]
)

# ==========================================
# STEP 29: COUNTY ID TO NAME MAPPING
# ==========================================

import json

with open("Data/county_id_to_name_map.json", "r") as file:
    county_mapping = json.load(file)

county_mapping_df = pd.DataFrame(
    list(county_mapping.items()),
    columns=["county_id", "county_name"]
)

print("\n--- County ID to Name Mapping ---")

print(county_mapping_df.head())

print("\nDataset Shape:")
print(county_mapping_df.shape)

print("\nColumns:")
print(county_mapping_df.columns.tolist())

print("\nData Types:")
print(county_mapping_df.dtypes)

print("\nMissing Values:")
print(county_mapping_df.isnull().sum())

print("\nDuplicate Rows:")
print(county_mapping_df.duplicated().sum())


# ==========================================
# STEP 30: REVEALED TARGETS DATA EXPLORATION
# ==========================================

revealed_targets_df = pd.read_csv(
    "Data/example_test_files/revealed_targets.csv"
)

print("\n--- Revealed Targets Dataset ---")

print(revealed_targets_df.head())

print("\nDataset Shape:")
print(revealed_targets_df.shape)

print("\nColumns:")
print(revealed_targets_df.columns.tolist())

print("\nData Types:")
print(revealed_targets_df.dtypes)

print("\nMissing Values:")
print(revealed_targets_df.isnull().sum())

print("\nDuplicate Rows:")
print(revealed_targets_df.duplicated().sum())




# ==========================================
# STEP 31: SAMPLE SUBMISSION EXPLORATION
# ==========================================

sample_submission_df = pd.read_csv(
    "Data/example_test_files/sample_submission.csv"
)

print("\n--- Sample Submission Dataset ---")

print(sample_submission_df.head())

print("\nDataset Shape:")
print(sample_submission_df.shape)

print("\nColumns:")
print(sample_submission_df.columns.tolist())

print("\nData Types:")
print(sample_submission_df.dtypes)

print("\nMissing Values:")
print(sample_submission_df.isnull().sum())

print("\nDuplicate Rows:")
print(sample_submission_df.duplicated().sum())



# ==========================================
# STEP 32: TEST DATASET EXPLORATION
# ==========================================

test_df = pd.read_csv(
    "Data/example_test_files/test.csv"
)

print("\n--- Test Dataset ---")

print(test_df.head())

print("\nDataset Shape:")
print(test_df.shape)

print("\nColumns:")
print(test_df.columns.tolist())

print("\nData Types:")
print(test_df.dtypes)

print("\nMissing Values:")
print(test_df.isnull().sum())

print("\nDuplicate Rows:")
print(test_df.duplicated().sum())



