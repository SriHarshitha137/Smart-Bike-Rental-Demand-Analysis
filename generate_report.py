"""
Automatic Academic Project Report Generator (.DOCX)
Project: Smart Bike Rental Demand Analysis and Prediction
File: generate_report.py
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

def create_report():
    doc = Document()

    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # ---------------------------------------------------------
    # TITLE PAGE
    # ---------------------------------------------------------
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title_p.add_run("SMART BIKE RENTAL DEMAND ANALYSIS AND PREDICTION USING MACHINE LEARNING\n")
    run_title.bold = True
    run_title.font.size = Pt(20)
    run_title.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = sub_p.add_run("A Mini Project Report Submitted in Partial Fulfillment of the Requirements for\nData Analytics and Visualization (DAV) Laboratory\n\n")
    run_sub.font.size = Pt(13)
    run_sub.italic = True

    meta_p = doc.add_paragraph()
    meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_p.add_run("Submitted By:\n").bold = True
    meta_p.add_run("[STUDENT NAME]\nRoll No: [ROLL NUMBER]\nBranch: Computer Science & Engineering (3rd Year B.Tech)\n\n")
    meta_p.add_run("Under the Guidance of:\n").bold = True
    meta_p.add_run("[FACULTY / SUPERVISOR NAME]\nDepartment of Computer Science & Engineering\n[COLLEGE / UNIVERSITY NAME]\nAcademic Year: 2026\n")

    doc.add_page_break()

    # ---------------------------------------------------------
    # CERTIFICATE & DECLARATION
    # ---------------------------------------------------------
    h_cert = doc.add_heading("CERTIFICATE OF ORIGINAL WORK", level=1)
    doc.add_paragraph(
        "This is to certify that the mini-project report entitled 'Smart Bike Rental Demand Analysis and Prediction "
        "Using Machine Learning' submitted by [STUDENT NAME] (Roll No: [ROLL NUMBER]) in partial fulfillment of "
        "the requirements for the award of Bachelor of Technology in Computer Science and Engineering is a bonafide "
        "record of work carried out under my supervision."
    )
    doc.add_paragraph("\n\nSignature of Faculty Guide: ____________________\nDate: October 2026\nDepartment of CSE")

    doc.add_page_break()

    # Helper function for adding structured sections
    def add_section(title, content, level=1):
        h = doc.add_heading(title, level=level)
        for p_text in content:
            p = doc.add_paragraph(p_text)
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(6)

    # ---------------------------------------------------------
    # 1. ABSTRACT
    # ---------------------------------------------------------
    add_section("1. Abstract", [
        "In modern urban transportation, bike-sharing networks offer an eco-friendly and affordable commuting alternative. "
        "However, operational success hinges on balancing bicycle supply with fluctuating user demand. This Data Analytics and "
        "Visualization (DAV) mini project presents an end-to-end data science and machine learning pipeline to analyze and "
        "forecast hourly bike rental demand using the Capital Bikeshare dataset (17,379 records).",
        "The project rigorously fulfills academic faculty requirements by conducting data preprocessing, wrangling, exploratory data "
        "analysis (EDA) across 10 distinct visualization techniques, and training two supervised regression models: Linear Regression "
        "(Ordinary Least Squares) and Random Forest Regressor. Target leakage was strictly prevented by excluding component counts "
        "('casual' and 'registered'). On an unseen 20% test split (3,476 records), the Random Forest Regressor demonstrated superior "
        "predictive power with an R² of 0.9081 and RMSE of 55.93 bikes, outperforming Linear Regression (R² = 0.3951, RMSE = 143.49). "
        "Finally, an interactive multi-page web application was developed in Streamlit, delivering an intuitive dashboard, analytical "
        "filters, and real-time demand prediction."
    ])

    # ---------------------------------------------------------
    # 2. INTRODUCTION & PROBLEM STATEMENT
    # ---------------------------------------------------------
    add_section("2. Introduction & Problem Statement", [
        "2.1 Problem Background: Rapid urban population growth has led to severe traffic congestion, elevated carbon emissions, and "
        "strained municipal transit. Docked bike-sharing systems address first-and-last-mile connectivity. However, demand varies wildly "
        "based on hourly commute schedules, weather anomalies, and seasonal shifts.",
        "2.2 Problem Statement: Bike-sharing operators face asymmetric station imbalances: stations in commercial zones suffer from empty docks "
        "during morning rush hours and dock overflows during evening return peaks. Lacking accurate demand forecasts, operators incur high logistics "
        "costs moving bikes via transport vans or lose riders to competing transit options.",
        "2.3 Objectives:\n"
        "• Ingest and verify a real-world dataset containing >= 1,000 records.\n"
        "• Perform systematic data cleaning, datetime transformation, and wrangling.\n"
        "• Generate meaningful visualizations discovering temporal and environmental usage drivers.\n"
        "• Implement, evaluate, and contrast at least two machine learning regression algorithms.\n"
        "• Deploy an interactive Streamlit web dashboard providing instant demand forecasting."
    ])

    # ---------------------------------------------------------
    # 3. DATASET DESCRIPTION
    # ---------------------------------------------------------
    add_section("3. Dataset Description & Verification", [
        "The project utilizes the actual 'hour.csv' dataset from the Capital Bikeshare system in Washington D.C., spanning the two-year period from 2011 to 2012. "
        "Direct inspection of the dataset revealed the following verified specifications:",
        "• Total Records: 17,379 rows (exceeds the faculty minimum of 1,000 records by over 17x).\n"
        "• Total Attributes: 17 columns.\n"
        "• Missing Values: Exactly 0 missing values across all columns.\n"
        "• Duplicate Rows: Exactly 0 duplicate records.\n"
        "• Target Variable: 'cnt' (Total rental bikes, ranging from 1 to 977 with a mean of 189.46 and standard deviation of 181.39)."
    ])

    # Dataset table
    table = doc.add_table(rows=1, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Light Shading Accent 1'
    hdr = table.rows[0].cells
    hdr[0].text = "Attribute"
    hdr[1].text = "Data Type"
    hdr[2].text = "Values / Range"
    hdr[3].text = "Description"

    cols_info = [
        ("instant", "Integer", "1 to 17,379", "Sequential record index"),
        ("dteday", "Date/String", "2011-01-01 to 2012-12-31", "Observation calendar date"),
        ("season", "Integer", "1: Spring, 2: Summer, 3: Fall, 4: Winter", "Categorical season indicator"),
        ("yr", "Integer", "0: 2011, 1: 2012", "Calendar year"),
        ("mnth", "Integer", "1 to 12", "Calendar month"),
        ("hr", "Integer", "0 to 23", "Hour of the day (24-hour format)"),
        ("holiday", "Binary", "0: No, 1: Yes", "Official public holiday indicator"),
        ("weekday", "Integer", "0 (Sun) to 6 (Sat)", "Day of the week"),
        ("workingday", "Binary", "1: Work Day, 0: Weekend/Holiday", "Commute activity indicator"),
        ("weathersit", "Integer", "1 (Clear) to 4 (Severe)", "Weather categorization"),
        ("temp", "Float", "0.02 to 1.00 (0°C to 41°C)", "Normalized ambient temperature"),
        ("atemp", "Float", "0.00 to 1.00 (0°C to 50°C)", "Normalized feeling temperature"),
        ("hum", "Float", "0.00 to 1.00 (0% to 100%)", "Normalized relative humidity"),
        ("windspeed", "Float", "0.00 to 0.85 (0 to 67 km/h)", "Normalized wind speed"),
        ("casual", "Integer", "0 to 367", "Unregistered casual rider count"),
        ("registered", "Integer", "0 to 886", "Registered subscriber count"),
        ("cnt", "Integer", "1 to 977", "TARGET: Total rental bike demand")
    ]
    for c_name, c_type, c_rng, c_desc in cols_info:
        row = table.add_row().cells
        row[0].text = c_name
        row[1].text = c_type
        row[2].text = c_rng
        row[3].text = c_desc

    doc.add_paragraph()

    # ---------------------------------------------------------
    # 4. PREPROCESSING & WRANGLING
    # ---------------------------------------------------------
    add_section("4. Data Preprocessing & Wrangling", [
        "4.1 Date Parsing: Converted string date column 'dteday' to standard Python datetime format for time-series extraction.",
        "4.2 Categorical Mapping: Generated human-readable labels for season, weather, weekday, workingday, and holiday to facilitate exploratory data visualization and user interfaces.",
        "4.3 Temperature & Weather De-normalization: Computed actual Celsius and percentage values from normalized attributes (e.g., temp * 41°C, hum * 100%) for realistic domain presentation.",
        "4.4 Data Leakage Prevention: In machine learning regression, target leakage occurs when predictor features contain direct knowledge of the target variable. Because 'cnt = casual + registered', including casual or registered riders would yield artificially inflated, useless models. Both variables were strictly removed from model feature sets.",
        "4.5 Data Wrangling: Aggregated bike demand by hour, month, season, and day type to uncover peak commute intervals and seasonal surges."
    ])

    # ---------------------------------------------------------
    # 5. EXPLORATORY DATA ANALYSIS & VISUALIZATION
    # ---------------------------------------------------------
    add_section("5. Exploratory Data Analysis (10 Core Visualizations)", [
        "1. Daily Rental Demand Line Chart: Depicts steady overall growth from 2011 to 2012, with recurring seasonal troughs during winter.",
        "2. Seasonal Rentals Bar Chart: Fall accounts for the highest average hourly volume (236.0 bikes/hr), followed by Summer (208.3 bikes/hr) and Winter (198.9 bikes/hr). Spring has the lowest (111.1 bikes/hr).",
        "3. Monthly Rentals Bar Chart: Mid-year months (June to September) consistently maintain peak averages above 230 bikes/hr.",
        "4. Hourly Rentals Bar Chart: Exhibits a sharp bimodal commuter pattern peaking at 8:00 AM (359 bikes/hr) and 5:00 PM (468 bikes/hr).",
        "5. Temperature Scatter Plot: Positive correlation (r = 0.40) demonstrating increased bike usage as temperatures warm toward 25°C - 30°C.",
        "6. Humidity Scatter Plot: Negative correlation (r = -0.32) indicating that muggy conditions and rainfall heavily suppress demand.",
        "7. Distribution Histogram & KDE: Highly right-skewed distribution (Mean = 189.46, Median = 142.00, Max = 977), standard for event count data.",
        "8. Seasonal Box Plot: Visualizes the wider interquartile range (IQR) and upper outliers during summer and fall months.",
        "9. Correlation Matrix Heatmap: Identifies 'hr' (0.39), 'temp' (0.40), and 'yr' (0.25) as the most influential numerical predictors.",
        "10. Working Day Comparison: Contrasts steep morning/evening office rush peaks on working days against broad afternoon leisure peaks on weekends."
    ])

    # ---------------------------------------------------------
    # 6. MACHINE LEARNING METHODOLOGY & COMPARISON
    # ---------------------------------------------------------
    add_section("6. Machine Learning Implementation & Model Comparison", [
        "6.1 Train-Test Split: Partitioned the 17,379 samples into an 80% training set (13,903 records) and a 20% testing set (3,476 records) using a fixed random seed (42).",
        "6.2 Model 1 - Linear Regression: Utilized Ordinary Least Squares (OLS) closed-form matrix solution theta = (X^T X)^(-1) X^T y. Linear Regression achieved an R² score of 0.3951 with an RMSE of 143.49 bikes.",
        "6.3 Model 2 - Random Forest Regressor: Implemented an ensemble of 15 randomized decision trees using bootstrap aggregation and random feature sub-selection. Random Forest captured intricate non-linear dependencies, achieving an R² score of 0.9081 and RMSE of 55.93 bikes."
    ])

    # ML Table
    ml_table = doc.add_table(rows=1, cols=5)
    ml_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    ml_table.style = 'Medium Shading 1 Accent 1'
    m_hdr = ml_table.rows[0].cells
    m_hdr[0].text = "Machine Learning Model"
    m_hdr[1].text = "MAE (Bikes)"
    m_hdr[2].text = "MSE"
    m_hdr[3].text = "RMSE (Bikes)"
    m_hdr[4].text = "R² Score"

    ml_rows = [
        ("Linear Regression (OLS)", "105.58", "20,590.23", "143.49", "0.3951 (39.5%)"),
        ("Random Forest Regressor", "35.69", "3,127.96", "55.93", "0.9081 (90.8%)")
    ]
    for r in ml_rows:
        cells = ml_table.add_row().cells
        for idx, text in enumerate(r):
            cells[idx].text = text

    doc.add_paragraph()

    # ---------------------------------------------------------
    # 7. WEB APPLICATION ARCHITECTURE
    # ---------------------------------------------------------
    add_section("7. Streamlit Web Application Architecture", [
        "The project includes a multi-page web application developed in Streamlit ('app.py'):\n"
        "• Module 1 - Dashboard: Displays 6 KPI metric cards and top hourly/seasonal distribution graphs.\n"
        "• Module 2 - Dataset Overview: Interactive table explorer, schema data types, missing value audit, and summary statistics.\n"
        "• Module 3 - Data Analysis: Filterable aggregations across season, month, day type, and weather.\n"
        "• Module 4 - Visualizations: Interactive rendering of all 10 Matplotlib/Seaborn figures with detailed insights.\n"
        "• Module 5 - ML Model Comparison: Side-by-side performance cards and visual bar charts comparing R² and RMSE.\n"
        "• Module 6 - Demand Prediction: Real-time prediction form allowing users to select hour, temperature, humidity, windspeed, and weather conditions, triggering live inference via the serialized Joblib models."
    ])

    # ---------------------------------------------------------
    # 8. CONCLUSION & REFERENCES
    # ---------------------------------------------------------
    add_section("8. Conclusion & Future Scope", [
        "Conclusion: This project successfully realized all academic objectives for the DAV Mini Project. By applying data preprocessing, wrangling, "
        "rigorous exploratory analysis, and supervised machine learning, the system provides accurate hourly bike demand forecasts. "
        "The Random Forest model demonstrated outstanding performance (R² = 0.9081), enabling smart mobility managers to optimize fleet rebalancing.",
        "Future Scope:\n"
        "• Real-time IoT station dock integration via REST APIs.\n"
        "• Deep learning sequence models (LSTM / GRU) for multi-hour forward horizon forecasting.\n"
        "• Mobile application for bike commuters showing predicted bike availability at nearby stations.",
        "References:\n"
        "1. Fanaee-T, Hadi, and Gama, Joao, 'Event labeling combining ensemble detectors and background knowledge', Progress in Artificial Intelligence (2014).\n"
        "2. Capital Bikeshare System Data, Washington D.C., USA.\n"
        "3. McKinney, Wes, 'Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython', O'Reilly Media.\n"
        "4. Pedregosa et al., 'Scikit-learn: Machine Learning in Python', JMLR 12, pp. 2825-2830."
    ])

    output_path = "Smart_Bike_Rental_Project_Report.docx"
    doc.save(output_path)
    print(f"[+] Successfully generated academic project report: {output_path}")

if __name__ == "__main__":
    create_report()
