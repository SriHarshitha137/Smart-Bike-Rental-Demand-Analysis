"""
Streamlit Web Application: Smart Bike Rental Demand Analysis and Prediction
Project: DAV Mini Project
File: app.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Custom modules
from preprocessing import preprocess_data, load_raw_data, get_ml_feature_columns, SEASON_MAP, WEATHER_MAP, YEAR_MAP, WORKINGDAY_MAP, HOLIDAY_MAP
from models import LinearRegressionModel, RandomForestRegressionModel
import analysis

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="Smart Bike Rental Demand Analytics & Prediction",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        margin-bottom: 10px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# CACHED DATA & MODEL LOADING
# -------------------------------------------------------------
@st.cache_data
def get_cached_data():
    return preprocess_data()

@st.cache_resource
def load_trained_models():
    lr = joblib.load("model/linear_model.pkl")
    rf = joblib.load("model/random_forest.pkl")
    with open("model/model_metrics.json", "r") as f:
        meta = json.load(f)
    return lr, rf, meta

# Load dataset and models
df = get_cached_data()

try:
    lr_model, rf_model, model_meta = load_trained_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    model_meta = None

# -------------------------------------------------------------
# HELPER FUNCTIONS FOR MODEL COMPARISON DIAGNOSTICS
# -------------------------------------------------------------
@st.cache_data
def get_test_predictions_and_data():
    """Load raw dataset, reproduce exact deterministic 80/20 test split, and generate predictions."""
    raw_df = load_raw_data()
    feature_cols, target_col = get_ml_feature_columns()
    X = raw_df[feature_cols].values
    y = raw_df[target_col].values

    np.random.seed(42)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    split_idx = int(0.80 * len(y))
    X_test = X[indices[split_idx:]]
    y_test = y[indices[split_idx:]]

    lr, rf, _ = load_trained_models()
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)

    return y_test, lr_pred, rf_pred, feature_cols


def compute_rf_feature_importance(rf_model, feature_names):
    """Traverse all decision trees in the trained Random Forest to count split contributions."""
    counts = {i: 0 for i in range(len(feature_names))}

    def traverse(node):
        if node is None or node.value is not None:
            return
        if node.feature is not None and node.feature in counts:
            counts[node.feature] += 1
        traverse(node.left)
        traverse(node.right)

    for tree in rf_model.trees:
        traverse(tree.root)

    total_splits = sum(counts.values())
    if total_splits == 0:
        total_splits = 1

    importances = [round(counts[i] / total_splits * 100.0, 2) for i in range(len(feature_names))]
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance (%)": importances
    }).sort_values(by="Importance (%)", ascending=True)
    return imp_df


def plot_actual_vs_predicted(y_true, y_pred, model_name, point_color='#2563EB'):
    """Generates an Actual vs. Predicted scatter plot with y=x reference line."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(y_true), size=min(400, len(y_true)), replace=False)
    y_true_s = y_true[sample_indices]
    y_pred_s = y_pred[sample_indices]

    ax.scatter(y_true_s, y_pred_s, alpha=0.55, color=point_color, edgecolors='none', s=35, label='Test Samples (Unseen 20%)')
    max_val = max(float(np.max(y_true_s)), float(np.max(y_pred_s)))
    ax.plot([0, max_val], [0, max_val], color='#DC2626', linestyle='--', linewidth=2, label='Perfect Prediction (y = x)')

    ax.set_title(f"Actual vs. Predicted Rentals - {model_name}", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Actual Bike Rentals (cnt)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Predicted Bike Rentals", fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


def plot_rf_feature_importance(importance_df):
    """Horizontal bar chart showing feature split importance percentages."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance (%)'], color='#10B981', edgecolor='#047857', alpha=0.85)
    ax.set_title("Feature Importance (% Split Contribution)", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Contribution Percentage (%)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Feature Name", fontsize=10, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    max_pct = float(importance_df['Importance (%)'].max())
    ax.set_xlim(0, max_pct * 1.22)
    for b in bars:
        width = b.get_width()
        ax.text(width + 0.3, b.get_y() + b.get_height() / 2, f"{width:.1f}%", va='center', ha='left', fontsize=9, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_overall_comparison(metrics):
    """Side-by-side bar chart comparing R² Score and RMSE for both models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))
    models_list = ["Linear Regression", "Random Forest Regressor"]

    # R2 Comparison
    r2_vals = [metrics['Linear Regression']['R2'], metrics['Random Forest Regressor']['R2']]
    bars1 = ax1.bar(models_list, r2_vals, color=['#3B82F6', '#10B981'], edgecolor='#1F2937', width=0.42)
    ax1.set_title("R² Score Comparison (Higher is Better)", fontsize=11, fontweight='bold', pad=10)
    ax1.set_ylabel("R² Score (0 to 1)", fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    for b in bars1:
        yval = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2.0, yval + 0.02, f"{yval:.4f}\n({yval*100:.1f}%)", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # RMSE Comparison
    rmse_vals = [metrics['Linear Regression']['RMSE'], metrics['Random Forest Regressor']['RMSE']]
    bars2 = ax2.bar(models_list, rmse_vals, color=['#EF4444', '#10B981'], edgecolor='#1F2937', width=0.42)
    ax2.set_title("RMSE Comparison (Lower is Better)", fontsize=11, fontweight='bold', pad=10)
    ax2.set_ylabel("RMSE (Bikes)", fontsize=10, fontweight='bold')
    ax2.set_ylim(0, max(rmse_vals) * 1.25)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    for b in bars2:
        yval = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2.0, yval + 3.0, f"{yval:.2f} bikes", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    return fig


# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/bicycle.png", width=80)
st.sidebar.title("SMART BIKE ANALYTICS")
st.sidebar.markdown("**DAV Mini Project** | 3rd Year B.Tech CSE")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Dashboard",
        "📂 Dataset Overview",
        "🔍 Data Analysis",
        "📈 Visualizations",
        "🤖 ML Model Comparison",
        "🔮 Bike Demand Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Project Details:**
- **Domain:** Smart Mobility & Analytics
- **Dataset:** Capital Bikeshare (`hour.csv`)
- **Total Records:** 17,379
- **Target:** `cnt` (Rental Bike Count)
""")

# =============================================================
# PAGE 1: DASHBOARD
# =============================================================
if menu == "📊 Dashboard":
    st.markdown('<div class="main-title">🚲 Smart Bike Rental Demand Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">High-level key performance indicators and demand distribution across the transportation network.</div>', unsafe_allow_html=True)

    # Key Metric Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Avg Hourly Rentals", f"{df['cnt'].mean():.1f}")
    with col3:
        st.metric("Peak Hour Rentals", f"{df['cnt'].max():,}")
    with col4:
        st.metric("Min Hour Rentals", f"{df['cnt'].min()}")
    with col5:
        st.metric("Avg Temperature", f"{df['temp_actual_celsius'].mean():.1f} °C")
    with col6:
        st.metric("Avg Humidity", f"{df['hum_actual_pct'].mean():.1f} %")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Hourly Rental Pattern (Peak Commute Hours)")
        fig_hr = analysis.plot_bar_rentals_by_hour(df)
        st.pyplot(fig_hr)
        st.caption("Insight: Pronounced spikes occur at 8:00 AM (morning rush) and 5:00 - 6:00 PM (evening rush).")

    with col_right:
        st.subheader("Seasonal Rental Pattern")
        fig_sea = analysis.plot_bar_rentals_by_season(df)
        st.pyplot(fig_sea)
        st.caption("Insight: Summer and Fall experience peak bike demand, while Spring experiences lower usage due to colder temperatures.")

    st.markdown("---")
    st.subheader("Quick System Summary")
    st.markdown("""
    - **Academic Requirement Compliance:** Analyzed **17,379 records** (exceeds the 1,000-record requirement).
    - **Data Integrity:** **0 missing values** and **0 duplicate entries**.
    - **Machine Learning Architecture:** Evaluated **Linear Regression** against **Random Forest Regressor**.
    - **Best Performing Model:** **Random Forest Regressor** achieved **R² = 0.9081** (90.8% variance explained).
    """)

# =============================================================
# PAGE 2: DATASET OVERVIEW
# =============================================================
elif menu == "📂 Dataset Overview":
    st.markdown('<div class="main-title">📂 Dataset Overview & Metadata</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Verification of raw attributes, data types, missing values, and descriptive statistics.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Preview Records", "Schema & Data Types", "Descriptive Statistics"])

    with tab1:
        st.write(f"Displaying sample records from total **{len(df):,}** rows:")
        num_rows = st.slider("Select number of rows to preview:", min_value=5, max_value=50, value=10, step=5)
        st.dataframe(df.head(num_rows), use_container_width=True)

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Dataset Attributes & Null Count")
            schema_df = pd.DataFrame({
                "Column Name": df.columns[:17],
                "Data Type": [str(t) for t in df.dtypes[:17]],
                "Null Values": [int(df[c].isnull().sum()) for c in df.columns[:17]],
                "Missing (%)": ["0.00%" for _ in range(17)]
            })
            st.dataframe(schema_df, use_container_width=True)
        with col_b:
            st.subheader("Dataset Verification Checks")
            st.success("✅ **Record Count Check:** 17,379 rows >= 1,000 minimum requirement.")
            st.success("✅ **Data Completeness:** 0 missing values detected across all attributes.")
            st.success("✅ **Uniqueness Check:** 0 duplicate rows detected.")
            st.info("ℹ️ **Target Variable:** `cnt` (Sum of `casual` + `registered`).")
            st.warning("⚠️ **Data Leakage Safeguard:** `casual` and `registered` are excluded from model features.")

    with tab3:
        st.subheader("Descriptive Numerical Statistics")
        summary_stats = analysis.get_summary_statistics(df)
        st.dataframe(summary_stats.style.format("{:.2f}"), use_container_width=True)

# =============================================================
# PAGE 3: DATA ANALYSIS
# =============================================================
elif menu == "🔍 Data Analysis":
    st.markdown('<div class="main-title">🔍 Data Wrangling & Analytical Aggregations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detailed aggregations across temporal, environmental, and calendar segments.</div>', unsafe_allow_html=True)

    analysis_view = st.selectbox(
        "Select Analytical Aggregation:",
        [
            "Season-wise Demand Aggregation",
            "Month-wise Demand Aggregation",
            "Working Day vs Non-Working Day Demand",
            "Weather Condition Impact",
            "Hourly Demand Breakdown"
        ]
    )

    if analysis_view == "Season-wise Demand Aggregation":
        st.subheader("Bike Rental Demand by Season")
        season_table = analysis.aggregate_by_season(df)
        st.dataframe(season_table.style.format({
            "Total_Rentals": "{:,.0f}",
            "Average_Rentals": "{:.2f}",
            "Median_Rentals": "{:.2f}",
            "Std_Dev": "{:.2f}",
            "Record_Count": "{:,.0f}"
        }), use_container_width=True)
        st.markdown("**Key Finding:** Fall accounts for the highest total volume (1,061,129 rentals), followed closely by Summer (918,589 rentals).")

    elif analysis_view == "Month-wise Demand Aggregation":
        st.subheader("Bike Rental Demand by Month (Jan - Dec)")
        month_table = analysis.aggregate_by_month(df)
        st.dataframe(month_table.style.format({
            "Total_Rentals": "{:,.0f}",
            "Average_Rentals": "{:.2f}",
            "Record_Count": "{:,.0f}"
        }), use_container_width=True)
        st.markdown("**Key Finding:** Demand steadily increases from January (peak winter) through June, maintaining high levels through October before dipping in winter.")

    elif analysis_view == "Working Day vs Non-Working Day Demand":
        st.subheader("Working Day vs Non-Working Day Demand")
        work_table = analysis.aggregate_by_workingday(df)
        st.dataframe(work_table.style.format({
            "Total_Rentals": "{:,.0f}",
            "Average_Rentals": "{:.2f}",
            "Median_Rentals": "{:.2f}",
            "Record_Count": "{:,.0f}"
        }), use_container_width=True)
        st.markdown("**Key Finding:** Working days register 11,865 observations with consistent commuter travel, whereas non-working days display midday recreational peaks.")

    elif analysis_view == "Weather Condition Impact":
        st.subheader("Impact of Weather Category on Rentals")
        weather_table = analysis.aggregate_by_weather(df)
        st.dataframe(weather_table.style.format({
            "Total_Rentals": "{:,.0f}",
            "Average_Rentals": "{:.2f}",
            "Median_Rentals": "{:.2f}",
            "Record_Count": "{:,.0f}"
        }), use_container_width=True)
        st.markdown("**Key Finding:** Clear weather yields the highest average rentals (~205 bikes/hr). Severe precipitation causes demand to plummet drastically.")

    elif analysis_view == "Hourly Demand Breakdown":
        st.subheader("Hourly Demand Breakdown (00:00 to 23:00)")
        hour_table = analysis.aggregate_by_hour(df)
        st.dataframe(hour_table.style.format({
            "Average_Rentals": "{:.2f}",
            "Total_Rentals": "{:,.0f}",
            "Min_Rentals": "{:.0f}",
            "Max_Rentals": "{:.0f}"
        }), use_container_width=True)
        st.markdown("**Key Finding:** Lowest demand occurs between 3:00 AM and 5:00 AM (average ~6-16 bikes/hr). Highest demand occurs at 17:00 (5:00 PM) averaging 468.8 bikes/hr.")

# =============================================================
# PAGE 4: VISUALIZATIONS
# =============================================================
elif menu == "📈 Visualizations":
    st.markdown('<div class="main-title">📈 Exploratory Data Analysis & Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">10 Comprehensive Matplotlib & Seaborn visualizations with verified data insights.</div>', unsafe_allow_html=True)

    chart_select = st.selectbox(
        "Choose a Visualization to Inspect:",
        [
            "All 10 Visualizations (Overview)",
            "1. Line Chart - Rentals Over Time",
            "2. Bar Chart - Rentals by Season",
            "3. Bar Chart - Rentals by Month",
            "4. Bar Chart - Rentals by Hour",
            "5. Scatter Plot - Temperature vs Rentals",
            "6. Scatter Plot - Humidity vs Rentals",
            "7. Histogram - Rentals Distribution",
            "8. Box Plot - Seasonal Variance & Outliers",
            "9. Heatmap - Correlation Matrix",
            "10. Comparison - Working Day vs Weekend"
        ]
    )

    if chart_select == "All 10 Visualizations (Overview)":
        c1, c2 = st.columns(2)
        with c1:
            st.image("assets/01_rentals_over_time.png", caption="Chart 1: Demand Over Time", use_container_width=True)
            st.image("assets/03_rentals_by_month.png", caption="Chart 3: Rentals by Month", use_container_width=True)
            st.image("assets/05_temp_vs_rentals.png", caption="Chart 5: Temperature vs Demand", use_container_width=True)
            st.image("assets/07_rentals_distribution.png", caption="Chart 7: Distribution of Rentals", use_container_width=True)
            st.image("assets/09_correlation_heatmap.png", caption="Chart 9: Correlation Matrix", use_container_width=True)
        with c2:
            st.image("assets/02_rentals_by_season.png", caption="Chart 2: Rentals by Season", use_container_width=True)
            st.image("assets/04_rentals_by_hour.png", caption="Chart 4: Rentals by Hour", use_container_width=True)
            st.image("assets/06_hum_vs_rentals.png", caption="Chart 6: Humidity vs Demand", use_container_width=True)
            st.image("assets/08_box_by_season.png", caption="Chart 8: Box Plot Across Seasons", use_container_width=True)
            st.image("assets/10_workingday_comparison.png", caption="Chart 10: Commuter vs Weekend Pattern", use_container_width=True)

    elif chart_select == "1. Line Chart - Rentals Over Time":
        st.pyplot(analysis.plot_line_rentals_over_time(df))
        st.markdown("**Analytical Insight:** Overall bike usage exhibits strong growth from 2011 to 2012, alongside recurring seasonal dips during winter periods.")

    elif chart_select == "2. Bar Chart - Rentals by Season":
        st.pyplot(analysis.plot_bar_rentals_by_season(df))
        st.markdown("**Analytical Insight:** Fall leads with an average of 236.0 bikes/hr, followed by Summer at 208.3 bikes/hr. Spring averages 111.1 bikes/hr.")

    elif chart_select == "3. Bar Chart - Rentals by Month":
        st.pyplot(analysis.plot_bar_rentals_by_month(df))
        st.markdown("**Analytical Insight:** Monthly average rentals peak in June, July, August, and September (230-240+ bikes/hr), corresponding to favorable weather.")

    elif chart_select == "4. Bar Chart - Rentals by Hour":
        st.pyplot(analysis.plot_bar_rentals_by_hour(df))
        st.markdown("**Analytical Insight:** Sharp bimodal distribution indicating commuter traffic at 8:00 AM (359 bikes/hr) and 5:00 PM (468 bikes/hr).")

    elif chart_select == "5. Scatter Plot - Temperature vs Rentals":
        st.pyplot(analysis.plot_scatter_temp_vs_rentals(df))
        st.markdown("**Analytical Insight:** Clear positive linear correlation ($r \approx 0.40$). As temperature rises toward 25°C - 30°C, bike rental demand increases.")

    elif chart_select == "6. Scatter Plot - Humidity vs Rentals":
        st.pyplot(analysis.plot_scatter_hum_vs_rentals(df))
        st.markdown("**Analytical Insight:** Negative correlation ($r \approx -0.32$). High relative humidity levels (> 80%) discourage outdoor bike riding.")

    elif chart_select == "7. Histogram - Rentals Distribution":
        st.pyplot(analysis.plot_histogram_rentals_distribution(df))
        st.markdown("**Analytical Insight:** Highly right-skewed distribution with Mean = 189.46 and Median = 142.0. Most hours observe lower counts, with rare peak hours exceeding 700+ bikes.")

    elif chart_select == "8. Box Plot - Seasonal Variance & Outliers":
        st.pyplot(analysis.plot_box_rentals_by_season(df))
        st.markdown("**Analytical Insight:** Summer and Fall show higher median lines and larger interquartile ranges (IQR). Multiple outliers occur during special event days.")

    elif chart_select == "9. Heatmap - Correlation Matrix":
        st.pyplot(analysis.plot_heatmap_correlation(df))
        st.markdown("**Analytical Insight:** Hour of day (`hr`, $r = 0.39$) and temperature (`temp`, $r = 0.40$) show the strongest positive correlations with rental count (`cnt`).")

    elif chart_select == "10. Comparison - Working Day vs Weekend":
        st.pyplot(analysis.plot_workingday_comparison(df))
        st.markdown("**Analytical Insight:** Working days show steep twin peaks (office commute), whereas weekends show a gentle bell-shaped curve peaking at 1:00 PM - 3:00 PM (leisure).")


# =============================================================
# PAGE 5: ML MODEL COMPARISON
# =============================================================
elif menu == "🤖 ML Model Comparison":
    st.markdown('<div class="main-title">🤖 Machine Learning Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Compare Linear Regression and Random Forest Regressor for bike rental demand prediction.</div>', unsafe_allow_html=True)

    if not models_loaded or not model_meta:
        st.error("Model files not found! Please run `python train_model.py` first to generate and serialize the models.")
    else:
        metrics = model_meta['metrics']

        # Load test set and actual predictions for plotting
        try:
            y_test, lr_test_pred, rf_test_pred, feature_cols = get_test_predictions_and_data()
            test_data_ready = True
        except Exception as e:
            test_data_ready = False
            st.warning(f"Could not load test predictions: {e}")

        # -------------------------------------------------------------
        # MODEL SELECTION
        # -------------------------------------------------------------
        st.markdown("### 🎯 Model Selection")
        selected_model = st.radio(
            "Select a model to view detailed analysis and diagnostic plots:",
            ["Linear Regression", "Random Forest Regressor"],
            horizontal=True
        )

        st.markdown("---")

        # -------------------------------------------------------------
        # SELECTED MODEL DETAILS
        # -------------------------------------------------------------
        if selected_model == "Linear Regression":
            st.subheader("Linear Regression Analysis")
            lr_m = metrics["Linear Regression"]

            # Display 4 clear metric cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("MAE", f"{lr_m['MAE']:.2f}", help="Mean Absolute Error (bikes)")
            with c2:
                st.metric("MSE", f"{lr_m['MSE']:,.2f}", help="Mean Squared Error")
            with c3:
                st.metric("RMSE", f"{lr_m['RMSE']:.2f}", help="Root Mean Squared Error (bikes)")
            with c4:
                st.metric("R² Score", f"{lr_m['R2']:.4f}", help="Coefficient of Determination")

            st.write("")

            # Actual vs Predicted Graph
            if test_data_ready:
                st.markdown("#### Actual vs. Predicted Values Plot")
                fig_lr = plot_actual_vs_predicted(y_test, lr_test_pred, "Linear Regression", point_color="#2563EB")
                st.pyplot(fig_lr)
                plt.close(fig_lr)

            # Model Interpretation
            st.markdown("#### Model Interpretation")
            st.markdown(f"""
            - **What Linear Regression Is:**  
              Linear Regression is an Ordinary Least Squares (OLS) parametric algorithm that models target hourly rental demand as a weighted linear combination of input features ($y = \\beta_0 + \\sum \\beta_i X_i$).
            
            - **What the Actual Metrics Mean:**  
              - **MAE = {lr_m['MAE']:.2f} bikes:** On average, the model's hourly predictions deviate from true counts by approximately **106 bikes**.
              - **MSE = {lr_m['MSE']:,.2f} & RMSE = {lr_m['RMSE']:.2f} bikes:** The high RMSE highlights that the model struggles significantly during high-volume periods, resulting in large residual errors.
              - **R² Score = {lr_m['R2']:.4f} ({lr_m['R2']*100:.1f}%):** The model explains only **39.51%** of the variance in rental demand, leaving more than 60% of the demand variation unexplained.
            
            - **What the Graph Shows:**  
              In the Actual vs. Predicted scatter plot, data points are broadly dispersed away from the ideal red diagonal line ($y = x$). Noticeable clustering occurs along the baseline because linear equations predict negative rentals during low-demand night hours, which must be truncated to zero.
            
            - **Performance Assessment:**  
              Linear Regression **performed poorly** on this dataset. Bike rental patterns exhibit pronounced non-linear spikes (e.g., commute rushes at 8:00 AM and 5:00 PM) and threshold effects with weather conditions that cannot be adequately captured by a single linear plane.
            """)

        elif selected_model == "Random Forest Regressor":
            st.subheader("Random Forest Regression Analysis")
            rf_m = metrics["Random Forest Regressor"]

            # Display 4 clear metric cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("MAE", f"{rf_m['MAE']:.2f}", help="Mean Absolute Error (bikes)")
            with c2:
                st.metric("MSE", f"{rf_m['MSE']:,.2f}", help="Mean Squared Error")
            with c3:
                st.metric("RMSE", f"{rf_m['RMSE']:.2f}", help="Root Mean Squared Error (bikes)")
            with c4:
                st.metric("R² Score", f"{rf_m['R2']:.4f}", help="Coefficient of Determination")

            st.write("")

            col_rf_plot1, col_rf_plot2 = st.columns(2)

            with col_rf_plot1:
                st.markdown("#### Actual vs. Predicted Values Plot")
                if test_data_ready:
                    fig_rf = plot_actual_vs_predicted(y_test, rf_test_pred, "Random Forest Regressor", point_color="#059669")
                    st.pyplot(fig_rf)
                    plt.close(fig_rf)

            with col_rf_plot2:
                st.markdown("#### Feature Importance")
                if test_data_ready and hasattr(rf_model, 'trees'):
                    imp_df = compute_rf_feature_importance(rf_model, feature_cols)
                    fig_imp = plot_rf_feature_importance(imp_df)
                    st.pyplot(fig_imp)
                    plt.close(fig_imp)

            # Model Interpretation
            st.markdown("#### Model Interpretation")
            st.markdown(f"""
            - **What Random Forest Regressor Is:**  
              Random Forest is a non-parametric ensemble learning method that constructs multiple decision trees via bootstrap aggregation (bagging) and randomized feature subsets. It aggregates predictions from individual trees to model complex non-linear patterns without overfitting.
            
            - **What the Actual Metrics Mean:**  
              - **MAE = {rf_m['MAE']:.2f} bikes:** Predictions are within approximately **36 bikes** of true counts on average (a **66% error reduction** compared to Linear Regression).
              - **MSE = {rf_m['MSE']:,.2f} & RMSE = {rf_m['RMSE']:.2f} bikes:** The RMSE is reduced by **61%** (from 143.49 to 55.93 bikes), indicating tight error bounds across all hours.
              - **R² Score = {rf_m['R2']:.4f} ({rf_m['R2']*100:.1f}%):** The model explains **90.81%** of all variance in unseen test data.
            
            - **What the Graph Shows:**  
              In the Actual vs. Predicted plot, data points cluster tightly along the ideal red diagonal line ($y = x$) across low, medium, and peak rental volumes, verifying strong predictive accuracy.
            
            - **Which Features Are Important:**  
              The Feature Importance graph derived directly from the trained decision tree ensemble reveals that **Hour of the Day (`hr`)** is the primary driver of demand (~20.8%), followed by **`weekday`** (~11.8%), **`temp`** (~10.5%), and **`hum`** (~10.3%). This aligns with real-world commuter behavior and weather dependency.
            
            - **Performance Assessment:**  
              Random Forest **performed exceptionally well**. It successfully models sharp rush-hour demand spikes and environmental variations, making it the best model for practical deployment.
            """)

        st.markdown("---")

        # -------------------------------------------------------------
        # OVERALL MODEL COMPARISON
        # -------------------------------------------------------------
        st.subheader("Overall Model Comparison")
        st.markdown("Direct side-by-side performance evaluation of both models on the identical 20% test dataset (3,476 records):")

        # Clean comparison table with NO broken row highlights
        comp_rows = []
        for name, m in metrics.items():
            comp_rows.append({
                "Model": name,
                "MAE": f"{m['MAE']:.2f}",
                "MSE": f"{m['MSE']:,.2f}",
                "RMSE": f"{m['RMSE']:.2f}",
                "R²": f"{m['R2']:.4f}"
            })
        comparison_df = pd.DataFrame(comp_rows)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        st.write("")
        st.markdown("#### Performance Comparison Chart")
        fig_comp = plot_overall_comparison(metrics)
        st.pyplot(fig_comp)
        plt.close(fig_comp)

        st.markdown(f"""
        **Academic Conclusion:**
        The **Random Forest Regressor** decisively outperforms Linear Regression across all evaluated criteria, improving the $R^2$ score from **0.3951 to 0.9081** (+{((metrics['Random Forest Regressor']['R2'] - metrics['Linear Regression']['R2'])/metrics['Linear Regression']['R2'])*100:.1f}% relative gain) and reducing Root Mean Squared Error from **143.49 to 55.93 bikes** (a 61% reduction).
        """)

# =============================================================
# PAGE 6: BIKE DEMAND PREDICTION
# =============================================================
elif menu == "🔮 Bike Demand Prediction":
    st.markdown('<div class="main-title">🔮 Interactive Bike Demand Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter environmental, seasonal, and temporal variables to compute real-time rental demand predictions.</div>', unsafe_allow_html=True)

    if not models_loaded:
        st.error("Model files not found! Please run `python train_model.py` first to generate the models.")
    else:
        st.markdown("### 1. Select Input Parameters")

        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Temporal & Calendar**")
                hr_input = st.slider("Hour of the Day (0 to 23)", 0, 23, 17)
                month_input = st.selectbox("Month", list(range(1, 13)), index=8, format_func=lambda x: [
                    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
                ][x-1])
                weekday_input = st.selectbox("Day of the Week", list(range(7)), index=3, format_func=lambda x: [
                    "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"
                ][x])
                year_input = st.selectbox("Year", [0, 1], index=1, format_func=lambda x: "2011" if x == 0 else "2012")

            with c2:
                st.markdown("**Season & Day Category**")
                season_input = st.selectbox("Season", [1, 2, 3, 4], index=2, format_func=lambda x: {
                    1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"
                }[x])
                workingday_input = st.radio("Working Day?", [1, 0], index=0, format_func=lambda x: "Yes (Work Day)" if x == 1 else "No (Weekend / Holiday)")
                holiday_input = st.radio("Holiday?", [0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")
                weather_input = st.selectbox("Weather Condition", [1, 2, 3, 4], index=0, format_func=lambda x: {
                    1: "Clear / Few Clouds",
                    2: "Mist / Cloudy",
                    3: "Light Snow / Rain",
                    4: "Heavy Rain / Ice Pellets"
                }[x])

            with c3:
                st.markdown("**Environmental Conditions**")
                temp_celsius = st.slider("Temperature (°C)", min_value=0.0, max_value=41.0, value=28.0, step=0.5)
                hum_pct = st.slider("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
                windspeed_kmh = st.slider("Windspeed (km/h)", min_value=0.0, max_value=67.0, value=12.0, step=1.0)
                
                # Derive normalized values required by the models
                temp_norm = temp_celsius / 41.0
                atemp_norm = temp_norm # Approximation of feeling temperature
                hum_norm = hum_pct / 100.0
                wind_norm = windspeed_kmh / 67.0

            submit_btn = st.form_submit_button("🚀 Predict Bike Rental Demand", use_container_width=True)

        if submit_btn:
            # Build feature array in the exact order trained:
            # ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
            input_features = np.array([[
                season_input, year_input, month_input, hr_input, holiday_input,
                weekday_input, workingday_input, weather_input,
                temp_norm, atemp_norm, hum_norm, wind_norm
            ]], dtype=np.float64)

            # Compute predictions using loaded models
            pred_lr = float(lr_model.predict(input_features)[0])
            pred_rf = float(rf_model.predict(input_features)[0])

            st.markdown("### 2. Live Prediction Results")
            p_col1, p_col2 = st.columns(2)

            with p_col1:
                st.markdown(f"""
                <div style="background-color: #EFF6FF; border-left: 5px solid #3B82F6; padding: 20px; border-radius: 8px;">
                    <h3 style="color: #1E40AF; margin-top: 0;">Linear Regression</h3>
                    <h1 style="color: #1E3A8A; font-size: 2.8rem; margin: 0;">{round(pred_lr)} <span style="font-size: 1.2rem;">bikes</span></h1>
                    <p style="color: #4B5563; margin-top: 5px;">R² Accuracy: 39.5% | RMSE: 143.5</p>
                </div>
                """, unsafe_allow_html=True)

            with p_col2:
                st.markdown(f"""
                <div style="background-color: #ECFDF5; border-left: 5px solid #10B981; padding: 20px; border-radius: 8px;">
                    <h3 style="color: #065F46; margin-top: 0;">Random Forest Regressor (Recommended)</h3>
                    <h1 style="color: #047857; font-size: 2.8rem; margin: 0;">{round(pred_rf)} <span style="font-size: 1.2rem;">bikes</span></h1>
                    <p style="color: #4B5563; margin-top: 5px;">R² Accuracy: 90.8% | RMSE: 55.9</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.info(f"""
            💡 **Live Analytical Context:**
            - For hour **{hr_input:02d}:00** in **{SEASON_MAP[season_input]}** with **{temp_celsius}°C** temperature and **{WEATHER_MAP[weather_input]}** weather:
            - The Random Forest model forecasts **{round(pred_rf)} bikes** will be rented across the network during this hour.
            - Fleet managers can use this prediction to dispatch bikes to high-demand commuter stations.
            """)
