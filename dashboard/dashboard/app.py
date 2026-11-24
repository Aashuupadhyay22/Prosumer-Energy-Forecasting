
# ==============================================================
# VOLT | PROSUMER ENERGY INTELLIGENCE
# Professional Cyberpunk Energy Forecasting Dashboard
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================
# 1. PROJECT PATHS
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "Processed_Data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "evaluation_results"
PREDICTION_DIR = PROJECT_ROOT / "predictions"

FEATURE_FILE = DATA_DIR / "feature_engineered_training.csv"
MODEL_FILE = MODEL_DIR / "energy_forecasting_model.pkl"
MODEL_FEATURES_FILE = MODEL_DIR / "model_features.json"
PREDICTION_FILE = PREDICTION_DIR / "test_predictions.csv"

# ==============================================================
# 2. PAGE CONFIGURATION
# ==============================================================

st.set_page_config(
    page_title="VOLT | Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================
# 3. PROFESSIONAL CYBERPUNK THEME
# ==============================================================

st.markdown("""
<style>

.stApp {
    background-color: #07121C;
    color: #EAF4FA;
}

[data-testid="stSidebar"] {
    background-color: #0B1D2A;
    border-right: 1px solid #17465B;
}

h1, h2, h3 {
    color: #EAF4FA !important;
}

p, label {
    color: #B8CCD8;
}

.metric-card {
    background: linear-gradient(135deg, #102B38, #10243B);
    border: 1px solid #20546A;
    border-radius: 13px;
    padding: 20px;
    min-height: 115px;
    box-shadow: 0 0 12px rgba(42, 210, 190, 0.07);
}

.metric-label {
    color: #7EDFC8;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
}

.metric-value {
    color: #FFFFFF;
    font-size: 29px;
    font-weight: 700;
    margin-top: 8px;
}

.metric-caption {
    color: #80A9BD;
    font-size: 12px;
    margin-top: 5px;
}

.section-card {
    background-color: #0D202D;
    border: 1px solid #1C4356;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 12px;
}

div[data-testid="stMetric"] {
    background-color: #102B38;
    border: 1px solid #20546A;
    border-radius: 12px;
    padding: 15px;
}

div[data-testid="stMetricLabel"] {
    color: #7EDFC8;
}

div[data-testid="stMetricValue"] {
    color: white;
}

.stButton button, .stDownloadButton button {
    background-color: #087F78;
    color: white;
    border: 1px solid #35D7B5;
    border-radius: 8px;
}

.stButton button:hover, .stDownloadButton button:hover {
    background-color: #0A9B8B;
    color: white;
    border: 1px solid #6CFFE0;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ==============================================================
# 4. PLOTLY THEME
# ==============================================================

CHART_TEMPLATE = "plotly_dark"

GREEN = "#39D6B0"
BLUE = "#55AFFF"
PURPLE = "#A78BFA"
ORANGE = "#F6B65B"

# ==============================================================
# 5. LOAD DATA
# ==============================================================

@st.cache_data
def load_training_data():

    df = pd.read_csv(FEATURE_FILE)

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["datetime", "target"]
    )

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    return df


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_FILE)

    with open(MODEL_FEATURES_FILE, "r") as file:
        features = json.load(file)

    return model, features


@st.cache_data
def load_predictions():

    if not PREDICTION_FILE.exists():
        return None

    return pd.read_csv(PREDICTION_FILE)


# ==============================================================
# 6. CHECK FILES
# ==============================================================

required_files = [
    FEATURE_FILE,
    MODEL_FILE,
    MODEL_FEATURES_FILE
]

missing_files = [
    str(file)
    for file in required_files
    if not file.exists()
]

if missing_files:

    st.error("Required project files are missing.")

    for file in missing_files:
        st.write(file)

    st.stop()

# ==============================================================
# 7. LOAD MODEL AND DATA
# ==============================================================

with st.spinner("Loading energy intelligence system..."):

    df = load_training_data()
    model, feature_columns = load_model()
    test_predictions = load_predictions()

# ==============================================================
# 8. GENERATE VALIDATION PREDICTIONS
# ==============================================================

@st.cache_data
def generate_validation_data():

    data = load_training_data()
    trained_model, features = load_model()

    X = data[
        [column for column in features if column in data.columns]
    ].copy()

    X = X.select_dtypes(
        include=[np.number]
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    split_position = int(len(data) * 0.80)

    X_valid = X.iloc[split_position:]

    actual = data["target"].iloc[split_position:].reset_index(drop=True)

    predictions = trained_model.predict(X_valid)

    validation = data.iloc[
        split_position:
    ].copy().reset_index(drop=True)

    validation["actual_target"] = actual
    validation["predicted_target"] = predictions

    validation["prediction_error"] = (
        validation["actual_target"]
        - validation["predicted_target"]
    )

    validation["absolute_error"] = (
        validation["prediction_error"].abs()
    )

    return validation


with st.spinner("Preparing forecasting analytics..."):
    validation = generate_validation_data()

# ==============================================================
# 9. HEADER
# ==============================================================

st.markdown("""
<div style="
    background: linear-gradient(110deg, #102B38, #10243B);
    border: 1px solid #20546A;
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 22px;
">
    <div style="color:#39D6B0; font-size:13px;
                letter-spacing:3px; font-weight:600;">
        VOLT / ENERGY INTELLIGENCE
    </div>
    <h1 style="margin:8px 0 4px 0; color:#FFFFFF;">
        Prosumer Energy Forecasting
    </h1>
    <div style="color:#82AFC3; font-size:14px;">
        Machine Learning • Energy Analytics • Predictive Intelligence
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================
# 10. SIDEBAR FILTERS
# ==============================================================

st.sidebar.markdown(
    "<h2 style='color:#39D6B0;'>⚡ CONTROL CENTER</h2>",
    unsafe_allow_html=True
)

st.sidebar.caption("Customize your energy analytics view")

min_date = validation["datetime"].min().date()
max_date = validation["datetime"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

filtered = validation.copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:

    start_date, end_date = date_range

    filtered = filtered[
        filtered["datetime"].dt.date.between(
            start_date,
            end_date
        )
    ]

if "county" in filtered.columns:

    counties = sorted(
        filtered["county"].dropna().unique().tolist()
    )

    selected_counties = st.sidebar.multiselect(
        "County",
        counties,
        default=counties
    )

    filtered = filtered[
        filtered["county"].isin(selected_counties)
    ]

if "is_business" in filtered.columns:

    business_options = sorted(
        filtered["is_business"].dropna().unique()
    )

    selected_business = st.sidebar.multiselect(
        "Business category",
        business_options,
        default=business_options,
        format_func=lambda x: (
            "Business" if x == 1 else "Non-business"
        )
    )

    filtered = filtered[
        filtered["is_business"].isin(selected_business)
    ]

if "is_consumption" in filtered.columns:

    consumption_options = sorted(
        filtered["is_consumption"].dropna().unique()
    )

    selected_type = st.sidebar.multiselect(
        "Energy type",
        consumption_options,
        default=consumption_options,
        format_func=lambda x: (
            "Consumption" if x == 1 else "Production"
        )
    )

    filtered = filtered[
        filtered["is_consumption"].isin(selected_type)
    ]

granularity = st.sidebar.selectbox(
    "Trend granularity",
    ["Hourly", "Daily", "Monthly"]
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **SYSTEM INFORMATION**

    Model: HistGradientBoostingRegressor

    Status: Validation model loaded

    Dashboard: Operational
    """
)

# ==============================================================
# 11. GLOBAL KPI CARDS
# ==============================================================

st.markdown("## OVERVIEW")

if filtered.empty:

    st.warning("No observations match the selected filters.")
    st.stop()

mae = mean_absolute_error(
    filtered["actual_target"],
    filtered["predicted_target"]
)

rmse = np.sqrt(
    mean_squared_error(
        filtered["actual_target"],
        filtered["predicted_target"]
    )
)

r2 = r2_score(
    filtered["actual_target"],
    filtered["predicted_target"]
) if len(filtered) > 1 else np.nan

total_rows = len(filtered)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">R² SCORE</div>
        <div class="metric-value">{r2:.4f}</div>
        <div class="metric-caption">Filtered validation performance</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">MAE</div>
        <div class="metric-value">{mae:.2f}</div>
        <div class="metric-caption">Mean absolute error</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">RMSE</div>
        <div class="metric-value">{rmse:.2f}</div>
        <div class="metric-caption">Root mean squared error</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">VALIDATION RECORDS</div>
        <div class="metric-value">{total_rows:,}</div>
        <div class="metric-caption">Records in current selection</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ==============================================================
# 12. NAVIGATION TABS
# ==============================================================

overview_tab, energy_tab, diagnostics_tab, explorer_tab = st.tabs([
    "⚡ Overview",
    "📊 Energy Analytics",
    "🧠 Model Diagnostics",
    "📁 Prediction Explorer"
])

# ==============================================================
# 13. OVERVIEW TAB
# ==============================================================

with overview_tab:

    st.subheader("Actual vs Predicted Energy")

    chart_data = filtered.sort_values(
        "datetime"
    )

    if granularity == "Hourly":

        chart_data = chart_data.copy()
        chart_data["period"] = chart_data["datetime"]

    elif granularity == "Daily":

        chart_data = (
            chart_data.set_index("datetime")
            .resample("D")[
                ["actual_target", "predicted_target"]
            ].mean()
            .reset_index()
        )

        chart_data["period"] = chart_data["datetime"]

    else:

        chart_data = (
            chart_data.set_index("datetime")
            .resample("MS")[
                ["actual_target", "predicted_target"]
            ].mean()
            .reset_index()
        )

        chart_data["period"] = chart_data["datetime"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_data["period"],
            y=chart_data["actual_target"],
            name="Actual",
            mode="lines",
            line=dict(color=GREEN, width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_data["period"],
            y=chart_data["predicted_target"],
            name="Predicted",
            mode="lines",
            line=dict(color=BLUE, width=2, dash="dash")
        )
    )

    fig.update_layout(
        template=CHART_TEMPLATE,
        height=420,
        xaxis_title="Date",
        yaxis_title="Energy target",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12)
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.caption(
        "Actual and predicted target values for the selected validation records."
    )

    st.subheader("Energy Type Comparison")

    if "is_consumption" in filtered.columns:

        energy_summary = (
            filtered.groupby("is_consumption")[
                ["actual_target", "predicted_target"]
            ].mean().reset_index()
        )

        energy_summary["Energy type"] = energy_summary[
            "is_consumption"
        ].map({
            0: "Production",
            1: "Consumption"
        })

        fig = px.bar(
            energy_summary,
            x="Energy type",
            y=["actual_target", "predicted_target"],
            barmode="group",
            color_discrete_sequence=[GREEN, BLUE],
            template=CHART_TEMPLATE,
            title="Average Production vs Consumption"
        )

        fig.update_layout(height=350)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==============================================================
# 14. ENERGY ANALYTICS TAB
# ==============================================================

with energy_tab:

    st.subheader("Hourly Energy Patterns")

    hourly = (
        filtered.groupby("hour")["actual_target"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        hourly,
        x="hour",
        y="actual_target",
        markers=True,
        color_discrete_sequence=[GREEN],
        template=CHART_TEMPLATE,
        title="Average Energy Target by Hour"
    )

    fig.update_layout(
        xaxis_title="Hour of day",
        yaxis_title="Average target",
        height=380
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    left, right = st.columns(2)

    with left:

        st.subheader("Weekly Energy Patterns")

        if "day_of_week" in filtered.columns:

            weekly = (
                filtered.groupby("day_of_week")[
                    "actual_target"
                ].mean().reset_index()
            )

            weekly["Day"] = weekly["day_of_week"].map({
                0: "Monday",
                1: "Tuesday",
                2: "Wednesday",
                3: "Thursday",
                4: "Friday",
                5: "Saturday",
                6: "Sunday"
            })

            fig = px.bar(
                weekly,
                x="Day",
                y="actual_target",
                color_discrete_sequence=[BLUE],
                template=CHART_TEMPLATE
            )

            fig.update_layout(height=350)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with right:

        st.subheader("Energy Distribution")

        fig = px.histogram(
            filtered,
            x="actual_target",
            nbins=40,
            color_discrete_sequence=[GREEN],
            template=CHART_TEMPLATE
        )

        fig.update_layout(
            xaxis_title="Actual energy target",
            yaxis_title="Frequency",
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Hourly and Weekly Heatmap")

    heatmap_data = filtered.pivot_table(
        index="day_of_week",
        columns="hour",
        values="actual_target",
        aggfunc="mean"
    )

    fig = px.imshow(
        heatmap_data,
        aspect="auto",
        color_continuous_scale=[
            "#10243B",
            "#087F78",
            "#39D6B0"
        ],
        template=CHART_TEMPLATE,
        labels={
            "x": "Hour",
            "y": "Day of week",
            "color": "Average target"
        }
    )

    fig.update_layout(height=400)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==============================================================
# 15. MODEL DIAGNOSTICS TAB
# ==============================================================

with diagnostics_tab:

    st.subheader("Actual vs Predicted Relationship")

    scatter = filtered.sample(
        n=min(5000, len(filtered)),
        random_state=42
    )

    fig = px.scatter(
        scatter,
        x="actual_target",
        y="predicted_target",
        opacity=0.45,
        color_discrete_sequence=[BLUE],
        template=CHART_TEMPLATE
    )

    lower = min(
        scatter["actual_target"].min(),
        scatter["predicted_target"].min()
    )

    upper = max(
        scatter["actual_target"].max(),
        scatter["predicted_target"].max()
    )

    fig.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            name="Ideal prediction",
            line=dict(color=GREEN, dash="dash")
        )
    )

    fig.update_layout(
        height=430,
        xaxis_title="Actual target",
        yaxis_title="Predicted target"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    left, right = st.columns(2)

    with left:

        st.subheader("Prediction Error Distribution")

        fig = px.histogram(
            filtered,
            x="prediction_error",
            nbins=50,
            color_discrete_sequence=[PURPLE],
            template=CHART_TEMPLATE
        )

        fig.update_layout(
            xaxis_title="Actual - Predicted",
            yaxis_title="Frequency",
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("Absolute Prediction Error")

        fig = px.box(
            filtered,
            y="absolute_error",
            color_discrete_sequence=[ORANGE],
            template=CHART_TEMPLATE
        )

        fig.update_layout(
            yaxis_title="Absolute error",
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Model Performance Summary")

    metrics_table = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Value": [
            round(mae, 4),
            round(rmse, 4),
            round(r2, 4) if pd.notna(r2) else np.nan
        ]
    })

    st.dataframe(
        metrics_table,
        use_container_width=True,
        hide_index=True
    )

# ==============================================================
# 16. PREDICTION EXPLORER TAB
# ==============================================================

with explorer_tab:

    st.subheader("Test Prediction Explorer")

    if test_predictions is not None:

        st.success(
            f"{len(test_predictions):,} test predictions loaded."
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total predictions",
            f"{len(test_predictions):,}"
        )

        c2.metric(
            "Mean predicted target",
            f"{test_predictions['target'].mean():.2f}"
        )

        c3.metric(
            "Maximum predicted target",
            f"{test_predictions['target'].max():.2f}"
        )

        st.subheader("Prediction Distribution")

        fig = px.histogram(
            test_predictions,
            x="target",
            nbins=50,
            color_discrete_sequence=[GREEN],
            template=CHART_TEMPLATE
        )

        fig.update_layout(
            xaxis_title="Predicted target",
            yaxis_title="Frequency",
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Prediction Records")

        search = st.text_input(
            "Search by row ID",
            placeholder="Enter a row ID..."
        )

        display_data = test_predictions.copy()

        if search and "row_id" in display_data.columns:

            display_data = display_data[
                display_data["row_id"].astype(str).str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

        st.dataframe(
            display_data.head(1000),
            use_container_width=True,
            hide_index=True
        )

        csv_data = test_predictions.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇ Download All Test Predictions",
            data=csv_data,
            file_name="test_predictions.csv",
            mime="text/csv"
        )

    else:

        st.warning("Test prediction file was not found.")

    st.divider()

    st.subheader("Validation Data Explorer")

    st.dataframe(
        filtered[
            [
                "datetime",
                "actual_target",
                "predicted_target",
                "prediction_error"
            ]
        ].head(500),
        use_container_width=True,
        hide_index=True
    )

    validation_download = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇ Download Filtered Validation Data",
        data=validation_download,
        file_name="filtered_validation_data.csv",
        mime="text/csv"
    )

# ==============================================================
# 17. FOOTER
# ==============================================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:20px;">
    <div style="color:#39D6B0; font-size:15px;
                font-weight:700; letter-spacing:2px;">
        VOLT | ENERGY INTELLIGENCE
    </div>
    <div style="color:#7895A7; font-size:12px; margin-top:8px;">
        Prosumer Energy Forecasting • Machine Learning Analytics
    </div>
</div>
""", unsafe_allow_html=True)

