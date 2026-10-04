# Smart Bike Rental Demand Analysis and Prediction Using Machine Learning

**Academic Course:** Data Analytics and Visualization (DAV) Mini Project  
**Student Level:** 3rd Year B.Tech Computer Science & Engineering (CSE)  
**Domain:** Transportation / Smart Mobility / Predictive Analytics  

---

## 📌 1. Project Overview & Faculty Requirements Compliance

This project solves the urban mobility problem of forecasting bike rental demand for the Capital Bikeshare bike-sharing network. It fulfills all requirements stipulated by the academic evaluation criteria:

| Requirement | Implementation & Proof | Status |
| :--- | :--- | :--- |
| **Dataset $\ge$ 1,000 records** | Analyzed **17,379 hourly records** from `hour.csv` | **PASSED** |
| **Data Preprocessing** | Datetime conversion, categorical decoding, actual temperature scaling | **PASSED** |
| **Data Wrangling** | Aggregations by season, month, hour, working day, and weather | **PASSED** |
| **Exploratory Data Analysis** | 10 visualizations (Line, Bar, Scatter, Hist, Box, Heatmap, Commuter) | **PASSED** |
| **At least 2 ML Models** | **Linear Regression (OLS)** & **Random Forest Regressor** | **PASSED** |
| **Data Leakage Safeguard** | `casual` and `registered` strictly excluded ($cnt = casual + registered$) | **PASSED** |
| **Model Comparison** | Evaluated via **MAE, MSE, RMSE, and R² Score** on identical 80/20 split | **PASSED** |
| **Model Persistence** | Models serialized and loaded using **Joblib** (`.pkl`) | **PASSED** |
| **Interactive Web Application** | Multi-page **Streamlit** dashboard with live user prediction | **PASSED** |

---

## 📊 2. Verified Evaluation Results

All numerical metrics were calculated directly on the 20% unseen test set (3,476 test records):

| Model | MAE (Bikes) | MSE | RMSE (Bikes) | R² Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 105.58 | 20,590.23 | 143.49 | 0.3951 (39.5%) |
| **Random Forest Regressor** | **35.69** | **3,127.96** | **55.93** | **0.9081 (90.8%)** |

> **Academic Conclusion:** The **Random Forest Regressor** outperforms Linear Regression by explaining **90.81%** of variance compared to 39.51%, reducing the root mean squared error by **61%** (from 143.49 to 55.93 bikes).

---

## 📁 3. Project Directory Structure

```text
Smart_Bike_Analytics/
│
├── dataset/
│   └── hour.csv               # Primary dataset (17,379 records, 17 columns)
│
├── model/
│   ├── linear_model.pkl       # Trained Linear Regression model
│   ├── random_forest.pkl      # Trained Random Forest Regressor
│   └── model_metrics.json     # Serialized evaluation metrics & metadata
│
├── assets/                    # Generated high-resolution visualization charts
│   ├── 01_rentals_over_time.png
│   ├── 02_rentals_by_season.png
│   ├── 03_rentals_by_month.png
│   ├── 04_rentals_by_hour.png
│   ├── 05_temp_vs_rentals.png
│   ├── 06_hum_vs_rentals.png
│   ├── 07_rentals_distribution.png
│   ├── 08_box_by_season.png
│   ├── 09_correlation_heatmap.png
│   ├── 10_workingday_comparison.png
│   └── 11_model_comparison.png
│
├── preprocessing.py           # Data ingestion, cleaning, and feature engineering
├── analysis.py                # Wrangling, aggregations, and plotting logic
├── models.py                  # Model architecture definitions (OLS & Tree Ensemble)
├── train_model.py             # 80/20 split, training, evaluation, and Joblib saving
├── app.py                     # Streamlit multi-page dashboard & prediction UI
├── verify_dataset.py          # Quick dataset verification check
├── requirements.txt           # Python dependencies
└── README.md                  # Project manual and viva guide
```

---

## 🚀 4. How to Run the Project (Windows & VS Code)

### Step 1: Open VS Code Terminal
Open PowerShell or Command Prompt in the project folder.

### Step 2: Run Dataset Verification
```powershell
.\venv\Scripts\python verify_dataset.py
```

### Step 3: Run Exploratory Data Analysis & Generate Visualizations
```powershell
.\venv\Scripts\python analysis.py
```
*(This generates and saves all 10 high-resolution charts into `assets/`)*

### Step 4: Train Machine Learning Models & Save `.pkl`
```powershell
.\venv\Scripts\python train_model.py
```
*(This trains both models, outputs the comparison table, and saves the `.pkl` files into `model/`)*

### Step 5: Launch the Streamlit Web Application
```powershell
.\venv\Scripts\python -m streamlit run app.py
```
*Your browser will automatically open at `http://localhost:8501`.*

---

## 🌐 5. Web Application Modules (Streamlit)

1. **📊 Dashboard:** Key KPIs (17,379 total records, 189.5 average hourly rentals, peak of 977 rentals) and quick visual summaries.
2. **📂 Dataset Overview:** Interactive table viewer, data types, missing value audit (0 missing), and 5-number descriptive statistics.
3. **🔍 Data Analysis:** Aggregations across seasons, months, hours, weather conditions, and working vs non-working days.
4. **📈 Visualizations:** Interactive review of the 10 Matplotlib/Seaborn figures with grounded domain insights.
5. **🤖 ML Model Comparison:** Side-by-side evaluation table and comparison chart (MAE, MSE, RMSE, R²).
6. **🔮 Bike Demand Prediction:** Interactive form with sliders for temperature, humidity, windspeed, hour, season, and weather. Generates instant predictions from both models.
