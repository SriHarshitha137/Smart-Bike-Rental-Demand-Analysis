# SMART BIKE RENTAL DEMAND ANALYSIS AND PREDICTION USING MACHINE LEARNING

**A Data Analytics & Visualization (DAV) Mini Project Report**  
**Course:** Data Analytics and Visualization Laboratory (3rd Year B.Tech CSE)  
**Domain:** Transportation / Smart Mobility / Predictive Analytics  

---

## 1. Project Title & Metadata
- **Project Title:** Smart Bike Rental Demand Analysis and Prediction Using Machine Learning
- **Student Name:** [Your Name]
- **Roll Number:** [Your Roll Number]
- **Branch / Year:** Computer Science & Engineering (3rd Year B.Tech)
- **Institution:** [Your College / University Name]
- **Faculty Supervisor:** [Your Faculty Name]
- **Academic Year:** 2026

---

## 2. Abstract
Bike-sharing systems play an indispensable role in contemporary green smart cities by solving first-and-last-mile public transportation challenges, mitigating carbon emissions, and reducing urban traffic congestion. However, an operational bottleneck faced by fleet managers is the high spatial-temporal fluctuation of rental demand: stations in commercial centers experience bike shortages during morning commuter hours and dock saturation in evening rush hours. 

This Data Analytics and Visualization (DAV) mini project presents a complete, rigorous data science and machine learning workflow to analyze and forecast hourly bike rental demand using the real-world Capital Bikeshare hourly dataset (17,379 records). The project adheres strictly to all academic faculty requirements: comprehensive data preprocessing and cleaning, exploratory data analysis across 10 distinct visualization techniques, implementation of two machine learning regression algorithms (**Linear Regression** via Ordinary Least Squares and **Random Forest Regressor**), and deployment of a feature-rich, interactive **Streamlit** web application. 

To maintain scientific rigor and prevent target leakage, component counts (`casual` and `registered`) were strictly excluded from predictor features. Evaluated on an unseen 20% test split (3,476 records), the Random Forest Regressor demonstrated outstanding predictive capability with an **R² Score of 0.9081 (90.81%)** and an **RMSE of 55.93 bikes**, vastly surpassing Linear Regression (**R² = 0.3951**, **RMSE = 143.49 bikes**). The accompanying Streamlit web application allows stakeholders to visualize analytical aggregations and perform live, real-time demand forecasting.

---

## 3. Problem Statement & Objectives
### 3.1 Problem Statement
In docked bike-sharing networks, bikes must be redistributed dynamically between pickup and drop-off stations to prevent dock saturation and starvation. Without accurate predictive models that incorporate weather conditions, commute schedules, and seasonal patterns, operators incur steep logistics expenses deploying diesel rebalancing vans or suffer rider churn due to unavailable bikes.

### 3.2 Objectives
1. **Fulfill Academic Threshold:** Ingest, inspect, and verify a real-world dataset containing over 1,000 observations (analyzed 17,379 records).
2. **Data Preprocessing & Wrangling:** Transform dates, decode categorical indices into intuitive semantic labels, and compute descriptive statistics.
3. **Exploratory Visualizations:** Construct at least 10 analytical plots revealing commuter peaks, weather sensitivity, and seasonal trends.
4. **Machine Learning Modeling:** Build, train, and evaluate two regression models on an identical 80/20 train-test partition without data leakage.
5. **Model Evaluation & Comparison:** Benchmark models using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.
6. **Web Application Deployment:** Provide an intuitive Streamlit interface featuring a KPI dashboard, dataset overview, interactive filters, model comparisons, and real-time inference.

---

## 4. Dataset Description & Verification
The dataset utilized is the hourly Capital Bikeshare dataset (`hour.csv`) from Washington D.C., covering the period from January 1, 2011 to December 31, 2012.

### 4.1 Dataset Statistics
- **Total Records (Rows):** 17,379 (meets and exceeds the faculty requirement of 1,000 records).
- **Total Attributes (Columns):** 17 attributes.
- **Missing / Null Values:** Exactly 0 missing values across all columns.
- **Duplicate Records:** Exactly 0 duplicate rows.
- **Target Variable:** `cnt` (Total hourly rental bike count).

### 4.2 Attribute Dictionary
| Attribute | Type | Values / Units | Description |
| :--- | :--- | :--- | :--- |
| `instant` | Integer | 1 to 17,379 | Sequential record index |
| `dteday` | Date/String | YYYY-MM-DD | Calendar date |
| `season` | Integer | 1: Spring, 2: Summer, 3: Fall, 4: Winter | Meteorological season |
| `yr` | Integer | 0: 2011, 1: 2012 | Year |
| `mnth` | Integer | 1 to 12 | Month of year |
| `hr` | Integer | 0 to 23 | Hour of day (24-hour clock) |
| `holiday` | Binary | 0: No, 1: Yes | Public holiday indicator |
| `weekday` | Integer | 0 (Sun) to 6 (Sat) | Day of week |
| `workingday` | Binary | 1: Work Day, 0: Weekend/Holiday | Commute schedule indicator |
| `weathersit` | Integer | 1: Clear, 2: Mist, 3: Light Rain/Snow, 4: Heavy Rain | Weather severity category |
| `temp` | Float | 0.02 to 1.00 ($t \times 41^\circ\text{C}$) | Normalized temperature |
| `atemp` | Float | 0.00 to 1.00 ($t \times 50^\circ\text{C}$) | Normalized feeling temperature |
| `hum` | Float | 0.00 to 1.00 ($h \times 100\%$) | Normalized relative humidity |
| `windspeed` | Float | 0.00 to 0.85 ($w \times 67\text{ km/h}$) | Normalized wind speed |
| `casual` | Integer | 0 to 367 | Non-registered casual user rentals |
| `registered` | Integer | 0 to 886 | Registered subscriber user rentals |
| `cnt` | Integer | 1 to 977 (Mean: 189.46) | **TARGET:** Total bike rentals |

---

## 5. Data Preprocessing & Data Wrangling
1. **Datetime Parsing:** Converted string timestamps (`dteday`) to standard pandas datetime objects.
2. **Label Decoding:** Mapped numerical integer codes to semantic strings (e.g., Season 1 $\rightarrow$ 'Spring', Weather 1 $\rightarrow$ 'Clear / Few Clouds') to improve clarity during analysis.
3. **De-normalization:** Scaled normalized values to intuitive metric units (°C, %, km/h) for user interaction.
4. **Target Leakage Safeguard:** Because `cnt = casual + registered`, casual and registered counts were strictly removed from the model feature set. Utilizing them would represent target leakage and invalidate real-world deployment.
5. **Feature Selection:** Selected 12 legitimate predictor features:
   `['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']`.

---

## 6. Exploratory Data Analysis & Visualizations (10 Core Charts)

1. **Daily Demand Over Time (Line Chart):** Illustrates sustained growth from 2011 to 2012 along with prominent winter seasonal troughs.
2. **Average Rentals by Season (Bar Chart):** Fall experiences peak average demand (236.0 bikes/hr), followed by Summer (208.3 bikes/hr), Winter (198.9 bikes/hr), and Spring (111.1 bikes/hr).
3. **Average Rentals by Month (Bar Chart):** June through September exhibit maximum utilization (> 230 bikes/hr), whereas January records the minimum (~94 bikes/hr).
4. **Average Rentals by Hour (Bar Chart):** Demonstrates pronounced commute spikes at 8:00 AM (359 bikes/hr) and 5:00 PM (468 bikes/hr).
5. **Temperature vs. Rentals (Scatter Plot with Trendline):** Positive correlation ($r = 0.40$), indicating ridership expands as temperatures rise toward comfortable levels (20°C - 30°C).
6. **Humidity vs. Rentals (Scatter Plot with Trendline):** Negative correlation ($r = -0.32$), indicating that high relative humidity and precipitation inhibit cycling.
7. **Rentals Distribution (Histogram & KDE):** Distinct right-skewed distribution (Mean: 189.46, Median: 142.00, Max: 977).
8. **Seasonal Box Plot:** Confirms higher medians and greater variability in Summer and Fall, with high-demand outliers during favorable holiday weekends.
9. **Pearson Correlation Heatmap:** Establishes hour of day (`hr`, $r = 0.39$) and temperature (`temp`, $r = 0.40$) as the two most impactful single features.
10. **Working Day vs. Non-Working Day Pattern:** Working days feature steep twin commute peaks, whereas weekends display a broad afternoon leisure plateau.

---

## 7. Machine Learning Modeling & Evaluation

### 7.1 Train-Test Partition
- **Total Records:** 17,379
- **Training Set (80%):** 13,903 records
- **Testing Set (20%):** 3,476 records (held out entirely from training)
- **Random Seed:** 42

### 7.2 Model 1: Linear Regression (Ordinary Least Squares)
- Closed-form algebraic solution: $\theta = (X^T X)^{-1} X^T y$.
- Fast and interpretable, but constrained by linear assumptions.
- Evaluation Metrics:
  - **MAE:** 105.58 bikes
  - **MSE:** 20,590.23
  - **RMSE:** 143.49 bikes
  - **R² Score:** 0.3951 (39.51% variance explained)

### 7.3 Model 2: Random Forest Regressor
- Ensemble of randomized regression trees with bootstrap aggregation (bagging) and random feature sub-selection.
- Naturally captures non-linear thresholds and complex multi-variable interactions.
- Evaluation Metrics:
  - **MAE:** 35.69 bikes
  - **MSE:** 3,127.96
  - **RMSE:** 55.93 bikes
  - **R² Score:** 0.9081 (90.81% variance explained)

### 7.4 Model Comparison Table
| Metric | Linear Regression | Random Forest Regressor | Performance Delta / Winning Model |
| :--- | :---: | :---: | :--- |
| **MAE** | 105.58 bikes | **35.69 bikes** | **66.2% error reduction** (Random Forest) |
| **MSE** | 20,590.23 | **3,127.96** | **84.8% reduction in squared error** |
| **RMSE** | 143.49 bikes | **55.93 bikes** | **61.0% lower root mean squared error** |
| **R² Score** | 0.3951 (39.5%) | **0.9081 (90.8%)** | **+129.8% relative gain in explanatory power** |

---

## 8. Web Application Architecture (Streamlit)
The web application (`app.py`) provides 6 interactive modules:
1. **Dashboard:** Displays KPI metric cards, high-level dataset statistics, and peak hour highlights.
2. **Dataset Overview:** Searchable dataframe viewer, data types, missing value audit, and summary statistics.
3. **Data Analysis:** Interactive slice-and-dice aggregations across season, month, day type, and weather conditions.
4. **Visualizations:** High-resolution rendering of all 10 Matplotlib/Seaborn figures with detailed domain takeaways.
5. **ML Model Comparison:** Comparative tables and bar charts highlighting the superiority of Random Forest.
6. **Bike Demand Prediction:** Interactive input form (sliders for hour, temp, humidity, windspeed, and dropdowns for weather/season) running real-time model inference.

---

## 9. Conclusion & Future Scope
- **Conclusion:** All faculty requirements were comprehensively fulfilled. The Random Forest Regressor demonstrated state-of-the-art predictive performance ($R^2 = 0.9081$), proving capable of modeling complex real-world mobility trends.
- **Future Scope:** Incorporation of live IoT weather sensor streams, integration with geographical GIS mapping (dock-level coordinates), and deep sequence modeling (LSTM) for multi-hour lookahead dispatching.
