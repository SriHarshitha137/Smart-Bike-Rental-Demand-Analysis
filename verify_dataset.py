"""
Step 1: Dataset Verification Script
Project: Smart Bike Rental Demand Analysis and Prediction
File: verify_dataset.py
"""

import os
import pandas as pd

def main():
    print("=" * 70)
    print("STEP 1: SMART BIKE RENTAL DATASET VERIFICATION")
    print("=" * 70)

    # 1. Determine file path
    csv_path = "dataset/hour.csv" if os.path.exists("dataset/hour.csv") else "hour.csv"
    print(f"[+] Loading dataset from: {csv_path}")

    # 2. Load dataset
    df = pd.read_csv(csv_path)
    print("[+] Dataset loaded successfully!")
    print("-" * 70)

    # 3. Exact Rows and Columns
    rows, cols = df.shape
    print(f"Total Number of Records (Rows) : {rows}")
    print(f"Total Number of Attributes (Cols): {cols}")
    print("-" * 70)

    # 4. Check Requirement (> 1,000 records)
    if rows >= 1000:
        print(f"[PASSED] Academic Requirement: Dataset has {rows:,} records (Requirement: >= 1,000 records).")
    else:
        print(f"[FAILED] Dataset has only {rows} records.")
    print("-" * 70)

    # 5. Column Names
    print("Exact Column Names in Dataset:")
    for idx, col in enumerate(df.columns, start=1):
        print(f"  {idx:2d}. {col}")
    print("-" * 70)

    # 6. Target Column Confirmation
    target_col = "cnt"
    if target_col in df.columns:
        print(f"[PASSED] Target Variable Identified: '{target_col}' (Total Rental Bike Count)")
    else:
        print(f"[FAILED] Target column '{target_col}' not found.")
    print("-" * 70)

    # 7. First 5 Rows Preview
    print("First 5 Rows Preview:")
    print(df.head())
    print("-" * 70)

    # 8. Data Types
    print("Column Data Types:")
    print(df.dtypes)
    print("-" * 70)

    # 9. Missing Values Check
    missing_vals = df.isnull().sum()
    print("Missing Values Per Column:")
    print(missing_vals)
    total_missing = missing_vals.sum()
    print(f"Total Missing Values in Dataset: {total_missing}")
    print("-" * 70)

    # 10. Duplicate Records Check
    duplicates = df.duplicated().sum()
    print(f"Duplicate Records in Dataset: {duplicates}")
    print("-" * 70)

    # 11. Summary Statistics of Target Variable
    print("Target Variable ('cnt') Summary Statistics:")
    print(f"  Count : {df['cnt'].count():,}")
    print(f"  Mean  : {df['cnt'].mean():.2f}")
    print(f"  Std   : {df['cnt'].std():.2f}")
    print(f"  Min   : {df['cnt'].min()}")
    print(f"  25%   : {df['cnt'].quantile(0.25):.2f}")
    print(f"  Median: {df['cnt'].median():.2f}")
    print(f"  75%   : {df['cnt'].quantile(0.75):.2f}")
    print(f"  Max   : {df['cnt'].max()}")
    print("=" * 70)
    print("[SUCCESS] Step 1 Verification Complete! Ready for Step 2.")
    print("=" * 70)

if __name__ == "__main__":
    main()
