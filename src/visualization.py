import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def plot_time_series(df, config):
    """Plot the time series data"""
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df[config['features']['target_col']])
    plt.title("Energy Consumption Time Series")
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("docs/time_series_plot.png")
    plt.close()

def plot_seasonality(df, config):
    """Plot seasonal patterns"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Hourly pattern
    hourly = df.groupby('hour')[config['features']['target_col']].mean()
    axes[0].plot(hourly.index, hourly.values)
    axes[0].set_title("Hourly Pattern")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Avg Consumption")
    
    # Daily pattern
    daily = df.groupby('day_of_week')[config['features']['target_col']].mean()
    axes[1].plot(daily.index, daily.values)
    axes[1].set_title("Daily Pattern (Day of Week)")
    axes[1].set_xlabel("Day of Week (0=Monday)")
    axes[1].set_ylabel("Avg Consumption")
    
    # Monthly pattern
    monthly = df.groupby('month')[config['features']['target_col']].mean()
    axes[2].plot(monthly.index, monthly.values)
    axes[2].set_title("Monthly Pattern")
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Avg Consumption")
    
    plt.tight_layout()
    plt.savefig("docs/seasonality_plots.png")
    plt.close()

def plot_model_comparison(results):
    """Plot model performance comparison"""
    metrics = list(results.values())[0].keys()
    models = list(results.keys())
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 8))
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        axes[i].bar(models, values)
        axes[i].set_title(metric.upper())
        axes[i].set_ylabel(metric)
    
    plt.tight_layout()
    plt.savefig("docs/model_comparison.png")
    plt.close()

def generate_visualizations():
    """Generate all visualizations"""
    config = load_config()
    df = pd.read_csv(config['data']['processed_path'], index_col=config['features']['timestamp_col'], parse_dates=True)
    
    # Ensure docs directory exists
    Path("docs").mkdir(exist_ok=True)
    
    # Generate plots
    plot_time_series(df, config)
    plot_seasonality(df, config)
    
    # Note: Model comparison plot requires model results
    # Typically run after modeling.py

if __name__ == "__main__":
    generate_visualizations()