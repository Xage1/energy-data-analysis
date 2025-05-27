import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_data():
    # Create date range
    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='h')
    
    # Base consumption pattern
    base = 100 + 20 * np.sin(2 * np.pi * dates.hour / 24)
    
    # Weekly seasonality
    weekly = 15 * np.sin(2 * np.pi * dates.dayofweek / 7)
    
    # Yearly seasonality
    yearly = 30 * np.sin(2 * np.pi * (dates.dayofyear + dates.hour / 24) / 365)
    
    # Random noise
    noise = np.random.normal(0, 5, len(dates))
    
    # Combine components
    consumption = base + weekly + yearly + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'energy_consumption': consumption
    })
    
    # Save to file
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/energy_data.csv", index=False)

if __name__ == "__main__":
    generate_sample_data()