#!/usr/bin/env python3
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class NaturalGasPricePredictor:
    def __init__(self, model_path="model/best_model.keras", data_path="Nat_Gas.csv"):
        self.model = tf.keras.models.load_model(model_path)
        self.data_path = data_path
        self.window_size = 12
        self.historical_data = self._load_historical_data()
        
    def _load_historical_data(self):
        df = pd.read_csv(self.data_path)
        df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
        df.set_index('Dates', inplace=True)
        return df['Prices']
    
    def predict_price_for_date(self, target_date):
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d')
        
        if target_date > self.historical_data.index.max():
            recent_data = self.historical_data.tail(self.window_size).values
        else:
            end_date = target_date - timedelta(days=1)
            window_data = self.historical_data[self.historical_data.index <= end_date]
            
            if len(window_data) < self.window_size:
                raise ValueError(f"Insufficient historical data")
            
            recent_data = window_data.tail(self.window_size).values
        
        X = recent_data.reshape(1, self.window_size, 1)
        prediction = self.model.predict(X, verbose=0)
        
        return float(prediction[0][0])

def main():
    if len(sys.argv) != 2:
        sys.exit(1)
    
    date_str = sys.argv[1]
    
    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        predictor = NaturalGasPricePredictor()
        predicted_price = predictor.predict_price_for_date(target_date)
        print(f"{predicted_price:.2f}")
        
    except:
        sys.exit(1)

if __name__ == "__main__":
    main() 