import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(filepath, train_size_ratio=0.7):
    # Load data
    df = pd.read_csv(filepath, delimiter='\t')
    df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread']
    
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Ensure 'Time' is in datetime.time format
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
    
    # Combine 'Date' and 'Time' into a single datetime column
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    
    # Feature Engineering: Extracting Hour and Minute, Cyclical Encoding
    df['Hour'] = df['Time'].apply(lambda x: x.hour)
    df['Minute'] = df['Time'].apply(lambda x: x.minute)
    
    # Cyclical encoding for Hour (0-23)
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    
    # Cyclical encoding for Minute (0-59)
    df['Minute_sin'] = np.sin(2 * np.pi * df['Minute'] / 60)
    df['Minute_cos'] = np.cos(2 * np.pi * df['Minute'] / 60)
    
    # Dropping unnecessary columns
    df.drop(['Hour', 'Minute', 'Vol', 'Time', 'Minute_sin', 'Minute_cos', 'DateTime'], axis=1, inplace=True)
    
    # Forward fill to handle NaNs
    df = df.ffill()
    # Drop remaining NaN rows
    df = df.dropna().reset_index(drop=True)
    
    # Before scaling, save unscaled prices for profit calculations
    df['Unscaled_Open'] = df['Open']
    df['Unscaled_High'] = df['High']
    df['Unscaled_Low'] = df['Low']
    df['Unscaled_Close'] = df['Close']
    df['Unscaled_Spread'] = df['Spread']  # Save unscaled spread
    
    # Exclude price columns and spread from scaling
    price_columns = ['Open', 'High', 'Low', 'Close', 'Unscaled_Open', 'Unscaled_High', 'Unscaled_Low', 'Unscaled_Close', 'Spread']
    
    # Select numerical columns to scale (excluding price columns and spread)
    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col not in price_columns]
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Fit scaler on the data and transform
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    # Determine the split index
    train_size = int(len(df) * train_size_ratio)
    
    # Split the data
    df_train = df.iloc[:train_size].reset_index(drop=True)
    df_test = df.iloc[train_size:].reset_index(drop=True)
    
    return df_train, df_test