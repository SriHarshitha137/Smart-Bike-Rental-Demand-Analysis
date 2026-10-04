"""
Data Wrangling, Exploratory Data Analysis (EDA) and Visualization Module
Project: Smart Bike Rental Demand Analysis and Prediction
File: analysis.py
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from preprocessing import preprocess_data, SEASON_MAP, WEATHER_MAP

# Set aesthetic styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. DATA WRANGLING & AGGREGATIONS
# -------------------------------------------------------------

def get_summary_statistics(df):
    """Calculate descriptive statistics for numerical columns."""
    cols = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
    summary = df[cols].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
    summary.rename(columns={'50%': 'median'}, inplace=True)
    return summary

def aggregate_by_season(df):
    """Aggregate total and average bike rentals by season."""
    grouped = df.groupby('season_label', observed=True)['cnt'].agg(
        Total_Rentals='sum',
        Average_Rentals='mean',
        Median_Rentals='median',
        Std_Dev='std',
        Record_Count='count'
    ).reset_index()
    # Sort order: Spring, Summer, Fall, Winter
    order = ['Spring', 'Summer', 'Fall', 'Winter']
    grouped['sort_key'] = grouped['season_label'].apply(lambda x: order.index(x) if x in order else 99)
    grouped = grouped.sort_values('sort_key').drop(columns=['sort_key'])
    return grouped

def aggregate_by_month(df):
    """Aggregate total and average bike rentals by month."""
    grouped = df.groupby(['mnth', 'month_label'], observed=True)['cnt'].agg(
        Total_Rentals='sum',
        Average_Rentals='mean',
        Record_Count='count'
    ).reset_index().sort_values('mnth')
    return grouped

def aggregate_by_hour(df):
    """Aggregate average rentals by hour of the day."""
    grouped = df.groupby('hr', observed=True)['cnt'].agg(
        Average_Rentals='mean',
        Total_Rentals='sum',
        Min_Rentals='min',
        Max_Rentals='max'
    ).reset_index().sort_values('hr')
    return grouped

def aggregate_by_workingday(df):
    """Aggregate rentals comparing working days vs non-working days."""
    grouped = df.groupby('workingday_label', observed=True)['cnt'].agg(
        Total_Rentals='sum',
        Average_Rentals='mean',
        Median_Rentals='median',
        Record_Count='count'
    ).reset_index()
    return grouped

def aggregate_by_weather(df):
    """Aggregate rentals across weather conditions."""
    grouped = df.groupby('weather_label', observed=True)['cnt'].agg(
        Total_Rentals='sum',
        Average_Rentals='mean',
        Median_Rentals='median',
        Record_Count='count'
    ).reset_index()
    return grouped

# -------------------------------------------------------------
# 2. REQUIRED VISUALIZATIONS (10 CHARTS)
# -------------------------------------------------------------

def plot_line_rentals_over_time(df, save=False):
    """Chart 1: Line Chart - Bike rentals aggregated daily over time."""
    daily = df.groupby('dteday', observed=True)['cnt'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(daily['dteday'], daily['cnt'], color='#1f77b4', linewidth=1.5, label='Daily Bike Rentals')
    ax.set_title("1. Daily Bike Rental Demand Over Time (2011 - 2012)", fontweight='bold')
    ax.set_xlabel("Date (Year-Month)")
    ax.set_ylabel("Total Daily Rental Count")
    ax.legend(loc='upper left')
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(ASSETS_DIR, "01_rentals_over_time.png"), dpi=200)
    return fig

def plot_bar_rentals_by_season(df, save=False):
    """Chart 2: Bar Chart - Average bike rentals by season."""
    season_agg = aggregate_by_season(df)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    bars = ax.bar(season_agg['season_label'], season_agg['Average_Rentals'], color=palette, edgecolor='black', alpha=0.85)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 3, f"{yval:.1f}", ha='center', va='bottom', fontweight='bold')
    ax.set_title("2. Average Hourly Bike Rentals by Season", fontweight='bold')
    ax.set_xlabel("Season")
    ax.set_ylabel("Average Hourly Rentals (cnt)")
    ax.set_ylim(0, max(season_agg['Average_Rentals']) * 1.15)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(ASSETS_DIR, "02_rentals_by_season.png"), dpi=200)
    return fig

def plot_bar_rentals_by_month(df, save=False):
    """Chart 3: Bar Chart - Average bike rentals by month."""
    month_agg = aggregate_by_month(df)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=month_agg, x='month_label', y='Average_Rentals', hue='month_label', ax=ax, palette='Blues_d', edgecolor='black', legend=False)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.0f}', (p.get_x() + p.get_width() / 2., height + 2),
                    ha='center', va='bottom', fontsize=9)
    ax.set_title("3. Average Hourly Bike Rentals by Month (Jan - Dec)", fontweight='bold')
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Hourly Rentals (cnt)")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(ASSETS_DIR, "03_rentals_by_month.png"), dpi=200)
    return fig

def plot_bar_rentals_by_hour(df, save=False):
    """Chart 4: Bar Chart - Average bike rentals by hour of day."""
    hour_agg = aggregate_by_hour(df)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = ['#ff7f0e' if h in [8, 17, 18] else '#1f77b4' for h in hour_agg['hr']]
    bars = ax.bar(hour_agg['hr'], hour_agg['Average_Rentals'], color=colors, edgecolor='black', alpha=0.85)
    ax.set_title("4. Average Bike Rentals by Hour of the Day (Peak Rush Hours Highlighted)", fontweight='bold')
    ax.set_xlabel("Hour of the Day (0 to 23)")
    ax.set_ylabel("Average Rentals (cnt)")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(ASSETS_DIR, "04_rentals_by_hour.png"), dpi=200)
    return fig

def plot_scatter_temp_vs_rentals(df, save=False):
    """Chart 5: Scatter Plot - Temperature vs bike rentals with trendline."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.regplot(data=df.sample(2500, random_state=42), x='temp_actual_celsius', y='cnt',
                scatter_kws={'alpha': 0.25, 'color': '#2ca02c'}, line_kws={'color': 'red', 'linewidth': 2}, ax=ax)
    ax.set_title("5. Relationship Between Temperature (°C) and Bike Rentals", fontweight='bold')
    ax.set_xlabel("Actual Temperature (°C)")
    ax.set_ylabel("Hourly Rental Count (cnt)")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(ASSETS_DIR, "05_temp_vs_rentals.png"), dpi=200)
    return fig

def plot_scatter_hum_vs_rentals(df, save=False):
    """Chart 6: Scatter Plot - Humidity vs bike rentals with trendline."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.regplot(data=df.sample(2500, random_state=42), x='hum_actual_pct', y='cnt',
                scatter_kws={'alpha': 0.25, 'color': '#9467bd'}, line_kws={'color': 'darkred', 'linewidth': 2}, ax=ax)
    ax.set_title("6. Relationship Between Humidity (%) and Bike Rentals", fontweight='bold')
    ax.set_xlabel("Actual Relative Humidity (%)")
    ax.set_ylabel("Hourly Rental Count (cnt)")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(ASSETS_DIR, "06_hum_vs_rentals.png"), dpi=200)
    return fig

def plot_histogram_rentals_distribution(df, save=False):
    """Chart 7: Histogram & KDE - Distribution of bike rental counts."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df['cnt'], bins=40, kde=True, color='#17becf', ax=ax, edgecolor='black')
    mean_val = df['cnt'].mean()
    median_val = df['cnt'].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f"Mean: {mean_val:.1f}")
    ax.axvline(median_val, color='green', linestyle='-', linewidth=1.5, label=f"Median: {median_val:.1f}")
    ax.set_title("7. Frequency Distribution of Total Bike Rentals (Right-Skewed)", fontweight='bold')
    ax.set_xlabel("Hourly Rental Count (cnt)")
    ax.set_ylabel("Frequency (Hour Observations)")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(ASSETS_DIR, "07_rentals_distribution.png"), dpi=200)
    return fig

def plot_box_rentals_by_season(df, save=False):
    """Chart 8: Box Plot - Bike rentals distribution by season."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = ['Spring', 'Summer', 'Fall', 'Winter']
    sns.boxplot(data=df, x='season_label', y='cnt', hue='season_label', order=order, palette='Set2', ax=ax, legend=False)
    ax.set_title("8. Box Plot: Bike Rentals Spread Across Seasons (Medians & Outliers)", fontweight='bold')
    ax.set_xlabel("Season")
    ax.set_ylabel("Hourly Rental Count (cnt)")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(ASSETS_DIR, "08_box_by_season.png"), dpi=200)
    return fig

def plot_heatmap_correlation(df, save=False):
    """Chart 9: Heatmap - Correlation matrix between numerical features."""
    cols = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'season', 'yr', 'cnt']
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5, ax=ax)
    ax.set_title("9. Pearson Correlation Matrix of Dataset Attributes", fontweight='bold')
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(ASSETS_DIR, "09_correlation_heatmap.png"), dpi=200)
    return fig

def plot_workingday_comparison(df, save=False):
    """Chart 10: Comparison - Hourly pattern on Working Days vs Weekends/Holidays."""
    hourly_work = df.groupby(['hr', 'workingday_label'], observed=True)['cnt'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.lineplot(data=hourly_work, x='hr', y='cnt', hue='workingday_label', marker='o', ax=ax, palette=['#e41a1c', '#377eb8'])
    ax.set_title("10. Hourly Demand Comparison: Working Days vs Non-Working Days", fontweight='bold')
    ax.set_xlabel("Hour of the Day (0 to 23)")
    ax.set_ylabel("Average Rental Count (cnt)")
    ax.set_xticks(range(0, 24))
    ax.legend(title="Day Type")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(ASSETS_DIR, "10_workingday_comparison.png"), dpi=200)
    return fig

def generate_all_charts():
    """Generates and saves all 10 required charts to assets/."""
    print("[*] Generating all 10 required visualizations...")
    df = preprocess_data()
    plot_line_rentals_over_time(df, save=True)
    plot_bar_rentals_by_season(df, save=True)
    plot_bar_rentals_by_month(df, save=True)
    plot_bar_rentals_by_hour(df, save=True)
    plot_scatter_temp_vs_rentals(df, save=True)
    plot_scatter_hum_vs_rentals(df, save=True)
    plot_histogram_rentals_distribution(df, save=True)
    plot_box_rentals_by_season(df, save=True)
    plot_heatmap_correlation(df, save=True)
    plot_workingday_comparison(df, save=True)
    print(f"[+] All 10 charts successfully saved to '{ASSETS_DIR}/' folder!")

if __name__ == "__main__":
    df = preprocess_data()
    print("=== SUMMARY STATISTICS ===")
    print(get_summary_statistics(df))
    print("\n=== AGGREGATION BY SEASON ===")
    print(aggregate_by_season(df))
    print("\n=== AGGREGATION BY WORKING DAY ===")
    print(aggregate_by_workingday(df))
    generate_all_charts()
