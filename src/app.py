import streamlit as st
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from src.modeling import arima_forecast, prophet_forecast, lstm_forecast
from src.visualization import plot_time_series, plot_seasonality

# Load config and data
@st.cache
def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

@st.cache
def load_data():
    config = load_config()
    df = pd.read_csv(config['data']['processed_path'], 
                    index_col=config['features']['timestamp_col'], 
                    parse_dates=True)
    return df, config

def main():
    st.title("Energy Consumption Forecasting Dashboard")
    
    df, config = load_data()
    target_col = config['features']['target_col']
    
    # Sidebar controls
    st.sidebar.header("Controls")
    show_raw_data = st.sidebar.checkbox("Show Raw Data", True)
    model_choice = st.sidebar.selectbox(
        "Select Model", 
        ["ARIMA", "Prophet", "LSTM"]
    )
    forecast_period = st.sidebar.slider(
        "Forecast Period (hours)", 
        min_value=24, max_value=168, value=72
    )
    
    # Main content
    st.header("Energy Consumption Overview")
    
    if show_raw_data:
        st.subheader("Raw Data")
        st.write(df.head())
    
    # Time series plot
    st.subheader("Time Series Plot")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df[target_col])
    ax.set_title("Energy Consumption Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Consumption")
    st.pyplot(fig)
    
    # Seasonal patterns
    st.subheader("Seasonal Patterns")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("Hourly Pattern")
        hourly = df.groupby('hour')[target_col].mean()
        st.line_chart(hourly)
    
    with col2:
        st.write("Daily Pattern")
        daily = df.groupby('day_of_week')[target_col].mean()
        st.line_chart(daily)
    
    with col3:
        st.write("Monthly Pattern")
        monthly = df.groupby('month')[target_col].mean()
        st.line_chart(monthly)
    
    # Model forecasting
    st.header("Model Forecasting")
    st.write(f"Using {model_choice} model to forecast next {forecast_period} hours")
    
    # Here you would add the actual forecasting code
    # For brevity, we'll simulate it
    if st.button("Run Forecast"):
        with st.spinner("Running forecast..."):
            # In a real app, you would call your forecasting functions here
            st.success("Forecast completed!")
            
            # Simulated forecast results
            forecast_dates = pd.date_range(
                start=df.index[-1], 
                periods=forecast_period+1, 
                freq='H'
            )[1:]
            
            # Simulated values - in real app use actual model predictions
            simulated_values = df[target_col].iloc[-forecast_period:].values + \
                            np.random.normal(0, 0.1, forecast_period)
            
            # Plot forecast
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df.index[-100:], df[target_col].iloc[-100:], label='Historical')
            ax.plot(forecast_dates, simulated_values, label='Forecast', color='orange')
            ax.set_title(f"{model_choice} Forecast")
            ax.legend()
            st.pyplot(fig)
            
            # Show forecast values
            st.subheader("Forecast Values")
            forecast_df = pd.DataFrame({
                'Timestamp': forecast_dates,
                'Forecasted Consumption': simulated_values
            })
            st.write(forecast_df)

if __name__ == "__main__":
    main()