"""
Machine Learning Training, Evaluation and Comparison Module
Project: Smart Bike Rental Demand Analysis and Prediction
File: train_model.py
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import load_raw_data, get_ml_feature_columns
from models import LinearRegressionModel, RandomForestRegressionModel

MODEL_DIR = "model"
ASSETS_DIR = "assets"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

def calculate_metrics(y_true, y_pred):
    """Calculates MAE, MSE, RMSE, R²."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    return {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    }

def train_and_save_all():
    print("=" * 70)
    print("STEP 7-11: MACHINE LEARNING MODEL TRAINING & COMPARISON")
    print("=" * 70)

    # 1. Load data
    df = load_raw_data()
    feature_cols, target_col = get_ml_feature_columns()

    X = df[feature_cols].values
    y = df[target_col].values

    print(f"[+] Loaded records: {X.shape[0]}")
    print(f"[+] Features ({len(feature_cols)}): {feature_cols}")
    print(f"[+] Target: {target_col}")
    print("[*] Note: 'casual' and 'registered' excluded to prevent data leakage (cnt = casual + registered).")
    print("-" * 70)

    # 2. Train-test split (80% train, 20% test)
    np.random.seed(42)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    split_idx = int(0.80 * len(y))

    X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
    y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]

    print(f"[+] Training Set : {X_train.shape[0]} records (80%)")
    print(f"[+] Testing Set  : {X_test.shape[0]} records (20%)")
    print("-" * 70)

    # ---------------------------------------------------------
    # MODEL 1: LINEAR REGRESSION
    # ---------------------------------------------------------
    print("[*] Training Model 1: Linear Regression (OLS)...")
    lr = LinearRegressionModel()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_metrics = calculate_metrics(y_test, lr_pred)

    print("  -> Linear Regression Evaluation Metrics:")
    for k, v in lr_metrics.items():
        print(f"     {k:4s} : {v}")
    print("-" * 70)

    # ---------------------------------------------------------
    # MODEL 2: RANDOM FOREST REGRESSOR
    # ---------------------------------------------------------
    print("[*] Training Model 2: Random Forest Regressor (Ensemble of Trees)...")
    rf = RandomForestRegressionModel(n_estimators=15, max_depth=10, max_features=8, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = calculate_metrics(y_test, rf_pred)

    print("  -> Random Forest Regressor Evaluation Metrics:")
    for k, v in rf_metrics.items():
        print(f"     {k:4s} : {v}")
    print("-" * 70)

    # ---------------------------------------------------------
    # 3. MODEL COMPARISON TABLE
    # ---------------------------------------------------------
    comparison_dict = {
        "Linear Regression": lr_metrics,
        "Random Forest Regressor": rf_metrics
    }
    comp_df = pd.DataFrame(comparison_dict).T
    print("=== MODEL EVALUATION & COMPARISON TABLE ===")
    print(comp_df)
    print("-" * 70)

    if rf_metrics['R2'] > lr_metrics['R2']:
        best_model = "Random Forest Regressor"
        improvement = ((rf_metrics['R2'] - lr_metrics['R2']) / lr_metrics['R2']) * 100
        print(f"[CONCLUSION] {best_model} is the superior model.")
        print(f"R² Score increased from {lr_metrics['R2']} to {rf_metrics['R2']} (+{improvement:.1f}% relative gain).")
        print(f"RMSE reduced from {lr_metrics['RMSE']} to {rf_metrics['RMSE']} bikes.")
    else:
        best_model = "Linear Regression"

    # ---------------------------------------------------------
    # 4. SAVE MODELS WITH JOBLIB
    # ---------------------------------------------------------
    lr_path = os.path.join(MODEL_DIR, "linear_model.pkl")
    rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    meta_path = os.path.join(MODEL_DIR, "model_metrics.json")

    joblib.dump(lr, lr_path)
    joblib.dump(rf, rf_path)

    metadata = {
        "features": feature_cols,
        "metrics": comparison_dict,
        "best_model": best_model,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0])
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[+] Successfully saved:")
    print(f"    - {lr_path}")
    print(f"    - {rf_path}")
    print(f"    - {meta_path}")
    print("-" * 70)

    # ---------------------------------------------------------
    # 5. GENERATE COMPARISON VISUALIZATION (Chart 11)
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    models_list = ["Linear Regression", "Random Forest"]

    # R2 Plot
    r2_vals = [lr_metrics['R2'], rf_metrics['R2']]
    bars1 = ax1.bar(models_list, r2_vals, color=['#3498db', '#2ecc71'], edgecolor='black', width=0.45)
    ax1.set_title("Model Comparison: R² Score (Higher is Better)", fontweight='bold')
    ax1.set_ylabel("R² Score (0 to 1)")
    ax1.set_ylim(0, 1.05)
    for b in bars1:
        yval = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

    # RMSE Plot
    rmse_vals = [lr_metrics['RMSE'], rf_metrics['RMSE']]
    bars2 = ax2.bar(models_list, rmse_vals, color=['#e74c3c', '#9b59b6'], edgecolor='black', width=0.45)
    ax2.set_title("Model Comparison: RMSE Error (Lower is Better)", fontweight='bold')
    ax2.set_ylabel("Root Mean Squared Error (cnt)", fontweight='bold')
    ax2.set_ylim(0, max(rmse_vals) * 1.18)
    for b in bars2:
        yval = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2.0, yval + 3.0, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(ASSETS_DIR, "11_model_comparison.png")
    fig.savefig(chart_path, dpi=200)
    print(f"[+] Model comparison chart saved to: {chart_path}")
    print("=" * 70)

    return lr, rf, comparison_dict

if __name__ == "__main__":
    train_and_save_all()
