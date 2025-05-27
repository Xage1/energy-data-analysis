CREATE TABLE energy_consumption (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    zone VARCHAR(32) NOT NULL,
    consumption_kwh FLOAT NOT NULL,
    fossil_percentage FLOAT,
    temperature FLOAT,
    humidity FLOAT,
    source VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE model_predictions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(64) NOT NULL,
    forecast_timestamp TIMESTAMPTZ NOT NULL,
    prediction_time TIMESTAMPTZ NOT NULL,
    horizon_hours INTEGER NOT NULL,
    predicted_kwh FLOAT NOT NULL,
    confidence_interval_lower FLOAT,
    confidence_interval_upper FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_energy_timestamp ON energy_consumption (timestamp);
CREATE INDEX idx_energy_zone ON energy_consumption (zone);
CREATE INDEX idx_predictions_model ON model_predictions(model_name);