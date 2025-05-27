# Energy Consumption Forecasting

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A machine learning project for forecasting energy consumption using time series analysis, with applications in telecom infrastructure management.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project demonstrates energy consumption forecasting using three different time series models:
- ARIMA (Classical statistical model)
- Prophet (Facebook's forecasting tool)
- LSTM (Deep learning approach)

The system helps telecom operators predict energy needs for base stations and data centers, enabling better capacity planning and cost optimization.

## API Integration Features

### Supported Data Sources
| API | Description | Authentication | Sample Config |
|-----|-------------|----------------|---------------|
| [EnergyData API](https://energydata.example.com) | Real-time grid data | API Key | `config.yaml` |
| [OpenWeatherMap](https://openweathermap.org/api) | Weather impact analysis | OAuth | `weather_config.yaml` |
| [Smart Meter API](https://developer.smartmeter.com) | Device-level consumption | JWT Token | `meter_config.yaml` |

## Features

- Data preprocessing and feature engineering pipeline
- Multiple forecasting models with comparative evaluation
- Interactive Streamlit dashboard for visualization
- Automated report generation
- Configuration management using YAML
- Synthetic data generation for testing

## Installation

1. **Clone the repository**:
   git clone https://github.com/yourusername/energy-consumption-forecasting.git
   cd energy-consumption-forecasting

2. **Create and activate virtual environment**:

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

3. **Install dependencies**:

pip install -r requirements.txt


## Usage

1. **Generate sample data**:

python gen_sample_data.py

2. **Process the data**:

bash
python src/data_processing.py

3. **Train models and evaluate**:

bash
python src/modeling.py

4. **Launch the dashboard**:

bash
streamlit run src/app.py

5. **Project Structure**
energy-data-analysis/
├── data/
│ ├── raw/ # Raw energy data files
│ └── processed/ # Cleaned and processed data
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ └── 02_model_experiments.ipynb
├── src/
│ ├── init.py
│ ├── data_processing.py # Data cleaning/feature engineering
│ ├── modeling.py # ARIMA, Prophet, LSTM models
│ ├── visualization.py # Plot generation
│ └── app.py # Streamlit dashboard
├── models/ # Saved model binaries
├── docs/ # Generated reports/visualizations
├── config.yaml # Configuration parameters
├── gen_sample_data.py # Synthetic data generator
├── requirements.txt # Python dependencies
└── README.md # This file

## Models Implemented
Model	Description	Best For
ARIMA	Classical time series model	Linear trends, stationary data
Prophet	Additive model with seasonality	Data with strong seasonal patterns
LSTM	Recurrent neural network	Complex patterns, long sequences


## Model Performance Results

| Model   | MAE (kW) | RMSE (kW) | MAPE (%) | Training Time (s) |
|---------|----------|-----------|----------|-------------------|
| ARIMA   | 12.4     | 15.2      | 8.3%     | 45                |
| Prophet | 10.1     | 13.7      | 6.9%     | 120               |
| LSTM    | 8.7      | 11.2      | 5.8%     | 300               |

Key:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error  
- **MAPE**: Mean Absolute Percentage Error

## Sample Visualizations

![Time Series Forecast](docs/forecast_example.png)
*Figure 1: Sample 72-hour forecast comparison*

![Seasonal Patterns](docs/seasonality_plots.png) 
*Figure 2: Hourly, daily and monthly consumption patterns*

## Contributing
Contributions are welcome! Please follow these steps:

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some amazing feature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

## License
Distributed under the MIT License. See LICENSE for more information.

**Project Maintainer**: Kenneth Kiarie Muketha