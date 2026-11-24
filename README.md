# ⚡ Prosumer Energy Forecasting

## 1. Project Overview

Prosumer Energy Forecasting is a machine learning project designed to predict energy consumption and production using historical energy data and related features.

The project uses Python, Pandas, NumPy, and Scikit-learn to explore data, preprocess datasets, engineer features, train a forecasting model, and generate predictions.

An interactive dashboard built with Streamlit presents model evaluation results and prediction outputs.

## 2. Project Objectives

- Explore and understand energy datasets.
- Clean and preprocess historical energy data.
- Perform feature engineering to improve forecasting.
- Train a machine learning regression model.
- Evaluate model performance using standard metrics.
- Generate predictions for test data.
- Visualize forecasting results through an interactive dashboard.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning |
| Joblib | Saving trained models |
| Streamlit | Interactive dashboard |
| VS Code | Development environment |

## 4. Project Structure

```text
Prosumer_Energy_Forecasting/
│
├── Data/
│   ├── train.csv
│   ├── client.csv
│   ├── electricity_prices.csv
│   ├── gas_prices.csv
│   ├── historical_weather.csv
│   └── example_test_files/
│
├── notebooks/
│   ├── 01_data_exploration.py
│   ├── 02_data_preprocessing.py
│   ├── 03_feature_engineering.py
│   ├── 04_model_training.py
│   ├── 05_model_evaluation.py
│   └── 06_generate_predictions.py
│
├── models/
│   ├── energy_forecasting_model.pkl
│   ├── model_features.json
│   └── model_evaluation.json
│
├── Processed_Data/
│   ├── processed_training_data.csv
│   └── feature_engineered_training.csv
│
├── evaluation_results/
│   ├── actual_vs_predicted.csv
│   ├── actual_vs_predicted.png
│   ├── prediction_scatter.png
│   ├── residual_distribution.png
│   └── evaluation_report.json
│
├── predictions/
│   └── test_predictions.csv
│
├── dashboard/
│   └── dashboard/
│       └── app.py
│
├── src/
│   └── main.py
│
├── requirements.txt
└── README.md
```

## 5. Project Workflow

### Step 1: Data Exploration
Explored the available datasets, examined columns, data types, missing values, duplicates, and target distributions.

### Step 2: Data Preprocessing
- Removed rows with missing target values.
- Converted date columns into datetime format.
- Investigated outliers and zero values.
- Merged client information.
- Integrated electricity and gas price information.
- Saved the processed training dataset.

### Step 3: Feature Engineering
Created additional features, including:
- Hour, day, month, and year.
- Day of the week and weekend indicators.
- Cyclical time features.
- Business and product interaction features.
- Historical lag features.
- Rolling mean and standard deviation features.

### Step 4: Model Training
Trained a HistGradientBoostingRegressor using Scikit-learn.

The dataset was divided chronologically into training and validation sets.

### Step 5: Model Evaluation
Evaluated the model using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Step 6: Test Predictions
Generated predictions for the supplied example test dataset and saved the output as a CSV file.

### Step 7: Interactive Dashboard
Developed a Streamlit dashboard to display:
- Model performance metrics.
- Actual versus predicted values.
- Prediction error distribution.
- Test prediction summaries.

## 6. Model Performance

The initial model evaluation produced the following results:

| Evaluation Metric | Result |
|---|---:|
| MAE | 35.3444 |
| RMSE | 131.2196 |
| R² Score | 0.9840 |

Model: HistGradientBoostingRegressor

**Note:** These are the current validation results. The validation setup uses historical target lag features, so performance may be optimistic compared with forecasting future periods without access to actual intermediate targets.

## 7. How to Run the Project

### Step 1: Clone or download the project

Open the project folder in Visual Studio Code.

### Step 2: Activate the virtual environment

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the project scripts

Run the Python scripts in the following order:

```bash
python notebooks/01_data_exploration.py
python notebooks/02_data_preprocessing.py
python notebooks/03_feature_engineering.py
python notebooks/04_model_training.py
python notebooks/05_model_evaluation.py
python notebooks/06_generate_predictions.py
```

### Step 5: Launch the dashboard

```bash
python -m streamlit run dashboard/dashboard/app.py
```

Open the local URL provided by Streamlit, usually:

`http://localhost:8501`

## 8. Output Files

The project generates:

- Processed training dataset.
- Feature-engineered training dataset.
- Trained machine learning model.
- Model feature configuration.
- Evaluation metrics and visualization files.
- Test prediction CSV file.
- Interactive forecasting dashboard.

## 9. Future Improvements

Potential future improvements include:

- Integrating weather forecasts and historical weather features.
- Comparing multiple machine learning algorithms.
- Improving time-series validation.
- Building recursive multi-step forecasting.
- Improving test-data feature preparation.
- Deploying the dashboard online.

## 10. Conclusion

This project demonstrates an end-to-end machine learning workflow for prosumer energy forecasting, from data exploration and preprocessing to model training, evaluation, test prediction, and dashboard visualization.

The project provides a foundation for further improvements in energy forecasting and data-driven energy management.

---

**Project:** Prosumer Energy Forecasting  
**Language:** Python  
**Model:** HistGradientBoostingRegressor  
**Dashboard:** Streamlit