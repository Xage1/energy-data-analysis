import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

def load_config():
    with open("config.yaml", "r" ) as f:
        config = yaml.safe_load(f)
    return config

def load_raw_data(config):
    #Load raw data from CSV
    df = pd.read_csv(config['data']['raw_path'])
    df[config['features']['timestamp_col']] = pd.to_datetime(df[config['features']['timestamp_col']])
    df.set_index(config['features']['timestamp_col'], inplace=True)
    return df
def clean_data(df, config):
     """ Clean and process Data """
     #Handle missing values
     df[config['features']['target_col']] = df[config['features']['target_col']].interpolate()

     #Remove outliers using IQR

     Q1 = df[config['features']['target_col']].quantile(0.25)
     Q3 = df[config['features']['target_col']].quantile(0.75)
     IQR = Q3 - Q1
     df = df[~((df[config['features']['target_col']] < (Q1 - 1.5 * IQR)) | (df[config['features']['target_col']] > (Q3 + 1.5 * IQR)))]
     return df

def create_features(df, config):

    for feature in config['features']['time_features']:
        if feature == "hour":
            df['hour'] = df.index.hour
        elif feature == "day_of_week":
            df['day_of_week'] = df.index.dayofweek
        elif feature == "month":
            df['month'] = df.index.month

    # Add lag features
    df['lag_24'] = df[config['features']['target_col']].shift(24)
    df['lag_168'] = df[config['features']['target_col']].shift(168)
    
    return df.dropna()

def process_data():
    config = load_config()

    #Ensure directories exist
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    #Load and process data
    df = load_raw_data(config)
    df = clean_data(df, config)
    df = create_features(df, config)

    #Saved processed data
    df.to_csv(config['data']['processed_path'])
    
    return df, config

if __name__ == "__main__":
    process_data()