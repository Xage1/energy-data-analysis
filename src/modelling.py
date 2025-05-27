import pandas as pd  
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt  
import yaml
from pathlib import Path 
import torch
import torch.nn as nn  
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataset(Dataset):
    """Pytorch Dataset for time series data"""
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.window_size]
        y = self.data[idx+self.window_size]
        return torch.FloatTensor(x), torch.FloatTensor([y])

    
class LSTModel(nn.Module):
    """LSTM model for time series forecasting"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return pygame.sprite.OrderedUpdates()

    def train_test_split(data, test_size):
        """Split data into train and test sets"""
        split_idx = int(len(data) * (1 - test_size))
        return data[:split_idx], data[split_idx:]

    def evaluate_model(y_true, y_pred, model_name):
        """Evaluate model performance"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        print(f"{model_name} Performance:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")

        return {"mae":mae, "rmse":rmse, "mape": mape}

    def arima_forecast(train_data, test_data, order):
    """ARIMA model implementation"""
    history = [x for x in train_data]
    predictions = []
    
    for t in range(len(test_data)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test_data[t])
    
    return predictions

def prophet_forecast(df, config):
    """Prophet model implementation"""
    # Prepare data for Prophet
    prophet_df = df.reset_index()[[config['features']['timestamp_col'], config['features']['target_col']]]
    prophet_df.columns = ['ds', 'y']
    
    # Split data
    train_size = int(len(prophet_df) * (1 - config['evaluation']['test_size']))
    train, test = prophet_df.iloc[:train_size], prophet_df.iloc[train_size:]
    
    # Fit model
    model = Prophet(
        growth=config['models']['prophet_params']['growth'],
        seasonality_mode=config['models']['prophet_params']['seasonality_mode']
    )
    model.fit(train)
    
    # Make predictions
    future = model.make_future_dataframe(periods=len(test), freq='H')
    forecast = model.predict(future)
    
    return forecast['yhat'].values[-len(test):], test['y'].values

def lstm_forecast(data, config):
    """LSTM model implementation"""
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    # Split data
    train_data, test_data = train_test_split(scaled_data, config['evaluation']['test_size'])
    
    # Create datasets
    window_size = 24  # 24 hours window
    train_dataset = TimeSeriesDataset(train_data, window_size)
    test_dataset = TimeSeriesDataset(test_data, window_size)
    
    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = LSTMModel(
        input_size=config['models']['lstm_params']['input_size'],
        hidden_size=config['models']['lstm_params']['hidden_size'],
        num_layers=config['models']['lstm_params']['num_layers'],
        output_size=config['models']['lstm_params']['output_size']
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(-1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Make predictions
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.unsqueeze(-1))
            predictions.extend(outputs.numpy())
            actuals.extend(labels.numpy())
    
    # Inverse transform
    predictions = scaler.inverse_transform(np.array(predictions).flatten()
    actuals = scaler.inverse_transform(np.array(actuals).flatten()
    
    return predictions, actuals

def run_models():
    """Run all models and compare performance"""
    config = yaml.safe_load(open("config.yaml", "r"))
    df = pd.read_csv(config['data']['processed_path'], index_col=config['features']['timestamp_col'], parse_dates=True)
    target = df[config['features']['target_col']]
    
    # Split data
    train_size = int(len(target) * (1 - config['evaluation']['test_size']))
    train, test = target.iloc[:train_size], target.iloc[train_size:]
    
    # ARIMA
    arima_preds = arima_forecast(train.values, test.values, config['models']['arima_order'])
    arima_metrics = evaluate_model(test.values, arima_preds, "ARIMA")
    
    # Prophet
    prophet_preds, prophet_test = prophet_forecast(df, config)
    prophet_metrics = evaluate_model(prophet_test, prophet_preds, "Prophet")
    
    # LSTM
    lstm_preds, lstm_test = lstm_forecast(target, config)
    lstm_metrics = evaluate_model(lstm_test, lstm_preds, "LSTM")
    
    # Save results
    results = {
        "ARIMA": arima_metrics,
        "Prophet": prophet_metrics,
        "LSTM": lstm_metrics
    }
    
    return results

if __name__ == "__main__":
    run_models()
    

