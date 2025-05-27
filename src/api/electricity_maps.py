# src/api/electricity_maps.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

class ElectricityMapsAPI:
    BASE_URL = "https://api.electricitymap.org/v3"
    
    def __init__(self):
        self.api_key = os.getenv("ELECTRICITY_MAPS_API_KEY")
        self.session = requests.Session()
        self.session.headers.update({"auth-token": self.api_key})
    
    def get_real_time_consumption(self, zone="US-CA"):
        """Get real-time energy consumption for a specific zone"""
        endpoint = f"{self.BASE_URL}/power-consumption-breakdown/latest"
        params = {"zone": zone}
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        return self._parse_consumption(response.json())
    
    def get_historical_data(self, zone="US-CA", days=30):
        """Get historical consumption data"""
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        
        endpoint = f"{self.BASE_URL}/power-consumption-breakdown/history"
        params = {
            "zone": zone,
            "start": start.isoformat() + "Z",
            "end": end.isoformat() + "Z"
        }
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        return self._parse_historical(response.json())
    
    def _parse_consumption(self, data):
        """Parse real-time response"""
        return {
            "timestamp": pd.to_datetime(data["datetime"]),
            "consumption": data["powerConsumptionTotal"],
            "zone": data["zone"],
            "fossil_free": data["fossilFreePercentage"]
        }
    
    def _parse_historical(self, data):
        """Parse historical data into DataFrame"""
        records = []
        for entry in data["history"]:
            records.append({
                "timestamp": pd.to_datetime(entry["datetime"]),
                "consumption": entry["powerConsumptionTotal"],
                "fossil_percentage": 100 - entry["fossilFreePercentage"]
            })
        return pd.DataFrame(records)