"""
Data Preprocessing Module
Project: Smart Bike Rental Demand Analysis and Prediction
File: preprocessing.py
"""

import os
import pandas as pd
import numpy as np

# Mappings for human-readable labels in EDA and Streamlit UI
SEASON_MAP = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
WEATHER_MAP = {
    1: "Clear / Few Clouds",
    2: "Mist / Cloudy",
    3: "Light Snow / Rain",
    4: "Heavy Rain / Ice Pellets"
}
YEAR_MAP = {0: 2011, 1: 2012}
WORKINGDAY_MAP = {0: "Non-Working Day (Weekend/Holiday)", 1: "Working Day"}
HOLIDAY_MAP = {0: "No", 1: "Yes"}
WEEKDAY_MAP = {
    0: "Sunday", 1: "Monday", 2: "Tuesday", 
    3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"
}
MONTH_MAP = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

def get_dataset_path():
    """Locate hour.csv in dataset/ or root directory."""
    if os.path.exists("dataset/hour.csv"):
        return "dataset/hour.csv"
    elif os.path.exists("hour.csv"):
        return "hour.csv"
    raise FileNotFoundError("Dataset hour.csv not found in dataset/ or current directory.")

def load_raw_data():
    """Load the raw dataset without transformations."""
    path = get_dataset_path()
    df = pd.read_csv(path)
    return df

def preprocess_data(df=None):
    """
    Load and preprocess the dataset:
    - Parses dteday into proper datetime format
    - Creates labeled categorical columns for descriptive analysis & plotting
    - Retains original numeric columns for ML training
    """
    if df is None:
        df = load_raw_data()
    
    # Make a copy to avoid chained assignment warnings
    data = df.copy()

    # Convert dteday to datetime
    data['dteday'] = pd.to_datetime(data['dteday'])

    # Create readable categorical labels for EDA
    data['season_label'] = data['season'].map(SEASON_MAP)
    data['weather_label'] = data['weathersit'].map(WEATHER_MAP)
    data['year_label'] = data['yr'].map(YEAR_MAP)
    data['workingday_label'] = data['workingday'].map(WORKINGDAY_MAP)
    data['holiday_label'] = data['holiday'].map(HOLIDAY_MAP)
    data['weekday_label'] = data['weekday'].map(WEEKDAY_MAP)
    data['month_label'] = data['mnth'].map(MONTH_MAP)

    # Actual denormalized temperature values (for intuitive understanding: temp * 41°C)
    data['temp_actual_celsius'] = data['temp'] * 41.0
    data['atemp_actual_celsius'] = data['atemp'] * 50.0
    data['hum_actual_pct'] = data['hum'] * 100.0
    data['windspeed_actual_kmh'] = data['windspeed'] * 67.0

    return data

def get_ml_feature_columns():
    """
    Returns the feature column list for model training.
    Excludes 'instant', 'dteday', 'casual', 'registered', 'cnt'
    to prevent target leakage (casual + registered = cnt).
    """
    features = [
        'season', 'yr', 'mnth', 'hr', 'holiday', 
        'weekday', 'workingday', 'weathersit', 
        'temp', 'atemp', 'hum', 'windspeed'
    ]
    target = 'cnt'
    return features, target

if __name__ == "__main__":
    print("[*] Testing preprocessing.py...")
    raw = load_raw_data()
    print(f"Loaded raw shape: {raw.shape}")
    processed = preprocess_data(raw)
    print(f"Processed shape: {processed.shape}")
    feat, targ = get_ml_feature_columns()
    print(f"Features ({len(feat)}): {feat}")
    print(f"Target: {targ}")
    print("[+] preprocessing.py verified successfully!")
