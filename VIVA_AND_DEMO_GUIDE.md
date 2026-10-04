# VIVA PREPARATION, DEMONSTRATION FLOW & SCREENSHOTS GUIDE

**Course:** Data Analytics and Visualization (DAV) Mini Project  
**Project:** Smart Bike Rental Demand Analysis and Prediction Using Machine Learning  

---

## 📸 1. Mandatory Screenshots Checklist for Your Report & Presentation

Capture the following 21 screenshots to include in your project report and viva presentation slides:

| # | What to Capture | Where to Find It |
| :---: | :--- | :--- |
| **1** | Terminal output of dataset verification | Run `.\venv\Scripts\python verify_dataset.py` |
| **2** | Dataset row count (17,379) and columns (17) | Terminal output or Streamlit "Dataset Overview" |
| **3** | Missing value check (0 missing across all 17 cols) | Terminal output or Streamlit "Dataset Overview" |
| **4** | Duplicate rows check (0 duplicates) | Terminal output of `verify_dataset.py` |
| **5** | Descriptive summary statistics table (`describe()`) | Streamlit "Dataset Overview" -> Tab 3 |
| **6** | Preprocessing execution output | Terminal running `preprocessing.py` |
| **7** | Data wrangling aggregated tables | Streamlit "Data Analysis" page |
| **8** | Chart 1: Daily Rentals Over Time | `assets/01_rentals_over_time.png` or Streamlit |
| **9** | Chart 2: Average Rentals by Season | `assets/02_rentals_by_season.png` or Streamlit |
| **10** | Chart 3: Rentals by Month (Jan - Dec) | `assets/03_rentals_by_month.png` or Streamlit |
| **11** | Chart 4: Hourly Rental Spikes (Commute Peaks) | `assets/04_rentals_by_hour.png` or Streamlit |
| **12** | Chart 5: Temperature vs Bike Demand | `assets/05_temp_vs_rentals.png` or Streamlit |
| **13** | Chart 6: Humidity vs Bike Demand | `assets/06_hum_vs_rentals.png` or Streamlit |
| **14** | Chart 7: Distribution Histogram & KDE | `assets/07_rentals_distribution.png` or Streamlit |
| **15** | Chart 8: Seasonal Box Plot & Outliers | `assets/08_box_by_season.png` or Streamlit |
| **16** | Chart 9: Pearson Correlation Matrix Heatmap | `assets/09_correlation_heatmap.png` or Streamlit |
| **17** | Chart 10: Working Day vs Weekend Demand Pattern | `assets/10_workingday_comparison.png` or Streamlit |
| **18** | ML Model Training terminal output | Terminal running `train_model.py` |
| **19** | Model Evaluation Comparison Table (MAE, MSE, RMSE, R²) | Streamlit "ML Model Comparison" |
| **20** | Model Comparison Bar Chart (R² & RMSE) | `assets/11_model_comparison.png` or Streamlit |
| **21** | Streamlit Live Prediction Form & Output Banners | Streamlit "Bike Demand Prediction" |

---

## 🎯 2. Step-by-Step 5–10 Minute Demonstration Flow

When demonstrating this project to your faculty or external examiner:

1. **Introduction (1 min):**
   - Introduce yourself and your project title: *"Smart Bike Rental Demand Analysis and Prediction Using Machine Learning"*.
   - State the objective: Predicting hourly bike rentals to assist urban fleet managers in rebalancing bike stations and meeting commuter demand.

2. **Dataset & Faculty Requirements Compliance (1 min):**
   - Highlight that the dataset contains **17,379 records** (comfortably exceeding the $\ge 1,000$ requirement).
   - Point out that there are **0 missing values** and **0 duplicate entries**.

3. **Data Preprocessing & Wrangling (1 min):**
   - Explain how datetime strings were parsed, categorical codes converted to readable labels, and normalized temperatures converted to Celsius.
   - **Crucial Point:** Emphasize that `casual` and `registered` counts were strictly excluded to eliminate **Target Leakage** ($cnt = casual + registered$).

4. **Exploratory Data Analysis & Visualizations (2 mins):**
   - Navigate to the **Visualizations** page on Streamlit.
   - Show **Chart 4 (Hourly Rentals)**: Explain the sharp commute rush hours at **8:00 AM** (morning office rush) and **5:00 PM** (evening return rush).
   - Show **Chart 10 (Working Day vs. Weekend)**: Explain how weekdays show commuter peaks while weekends show leisure midday peaks.
   - Show **Chart 5 & 6**: Explain the positive impact of temperature and negative impact of humidity/rain.

5. **Machine Learning Models & Evaluation (2 mins):**
   - Navigate to **ML Model Comparison**.
   - Explain the 80/20 train-test split (13,903 training rows, 3,476 testing rows).
   - Present the comparison table:
     * **Linear Regression:** R² = 0.3951, RMSE = 143.49 bikes.
     * **Random Forest Regressor:** R² = 0.9081, RMSE = 55.93 bikes.
   - Explain why Random Forest is superior: It captures non-linear relationships (e.g., commute spikes and storm drop-offs) that linear models cannot fit.

6. **Live Demand Prediction Demo (2 mins):**
   - Navigate to **Bike Demand Prediction**.
   - Set the inputs: **5:00 PM**, **Fall Season**, **28°C Temperature**, **Clear Weather**, **Working Day**.
   - Click **"🚀 Predict Bike Demand"**.
   - Show the live forecast (~800 bikes predicted by Random Forest).
   - Explain how a fleet manager would use this prediction to dispatch bikes to busy subway/office stations in advance.

---

## 💡 3. Comprehensive Viva Questions & Beginner-Friendly Answers

### Section A: Dataset & Preprocessing
**Q1: What is the dataset used, and what does each row represent?**  
**A:** The dataset is the Capital Bikeshare hourly dataset (`hour.csv`) from Washington D.C. Each row represents one single hour of bike rental operations, containing weather conditions, temporal information, and the total count of rented bikes.

**Q2: What is the target variable?**  
**A:** The target variable is `cnt` (count), representing the total number of bicycles rented in that specific hour.

**Q3: What is Data Preprocessing, and why did you perform it?**  
**A:** Preprocessing prepares raw, messy data for statistical analysis and machine learning. In our project, we converted string date columns into datetime format, mapped numeric codes into readable categories (e.g., Season 1 to 'Spring'), and verified that there are no missing or duplicate values.

**Q4: What is Target Leakage (Data Leakage), and how did you prevent it?**  
**A:** Target leakage occurs when predictor features contain information that would not be available at prediction time or directly reveal the target. In our dataset, $cnt = casual + registered$. If we included `casual` or `registered` as input features, the model would simply add them together and achieve trivial 100% accuracy, but it would be completely useless in real life because those counts are unknown before the hour ends. We strictly excluded both columns from our feature set.

---

### Section B: Exploratory Data Analysis & Visualizations
**Q5: What is Exploratory Data Analysis (EDA)?**  
**A:** EDA is the process of examining, summarizing, and visualizing a dataset to uncover patterns, anomalies, test hypotheses, and verify assumptions before building predictive models.

**Q6: What did your correlation heatmap show?**  
**A:** The Pearson correlation heatmap revealed that the hour of the day (`hr`, $r = 0.39$) and temperature (`temp`, $r = 0.40$) have the highest positive correlations with rental demand, while humidity (`hum`, $r = -0.32$) has a significant negative correlation.

**Q7: Why does the hourly demand plot show two distinct peaks on working days?**  
**A:** Because bikes in urban areas are predominantly used by daily office commuters. The first peak occurs around 8:00 AM (travel to work), and the second larger peak occurs around 5:00 PM to 6:00 PM (travel home from work).

---

### Section C: Machine Learning & Evaluation
**Q8: Why did you split the dataset into 80% train and 20% test sets?**  
**A:** If we evaluate a model on the same data it was trained on, it might simply memorize the data (overfitting). Splitting into an 80% training set and a 20% unseen test set allows us to measure how well the model generalizes to new, real-world data.

**Q9: Explain Linear Regression and its mathematical foundation.**  
**A:** Linear Regression models the relationship between independent features $X$ and the continuous target $y$ as a straight line: $y = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n$. We solved it using Ordinary Least Squares (OLS): $\theta = (X^T X)^{-1} X^T y$, which finds the weights that minimize the sum of squared differences between actual and predicted values.

**Q10: Explain Random Forest Regressor and why it performed better than Linear Regression.**  
**A:** Random Forest is an ensemble learning method that builds multiple decision trees using bootstrap aggregation (bagging) and random feature sub-selection. It performed significantly better (R² of 90.8% vs 39.5%) because bike rental demand is highly non-linear (e.g., sudden spikes at 8 AM and 5 PM, or steep drop-offs during rain), which linear models cannot capture.

**Q11: What do MAE, MSE, RMSE, and R² Score mean?**  
- **MAE (Mean Absolute Error):** The average absolute difference between actual and predicted bikes. In our Random Forest model, MAE is only 35.69 bikes.
- **MSE (Mean Squared Error):** The average of squared errors, heavily penalizing large mistakes.
- **RMSE (Root Mean Squared Error):** The square root of MSE, expressed in the same units as the target (bikes). For Random Forest, RMSE is 55.93 bikes.
- **R² Score (Coefficient of Determination):** The proportion of variance in the target variable explained by the model. 0.9081 means our model explains 90.81% of the variation in bike rentals.

---

### Section D: Web Application & Deployment
**Q12: Why did you choose Streamlit for the web application?**  
**A:** Streamlit is an open-source Python framework designed specifically for data science and machine learning applications. It allows creating responsive, interactive dashboards directly in Python without needing complex frontend frameworks like React or Angular.

**Q13: How does the Prediction page work in your web app?**  
**A:** When a user selects environmental and calendar values in the web form and clicks "Predict Bike Demand", the app passes the user inputs to our trained model (`model/random_forest.pkl` loaded via Joblib) and calculates the forecasted bike demand in real time.
