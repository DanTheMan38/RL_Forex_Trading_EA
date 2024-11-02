import pandas as pd
import ta

def add_indicators(df):
    # 1. Simple Moving Averages (SMA)
    df['SMA_10'] = ta.trend.SMAIndicator(close=df['Close'], window=10).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()

    # 2. Exponential Moving Averages (EMA)
    df['EMA_12'] = ta.trend.EMAIndicator(close=df['Close'], window=12).ema_indicator()
    df['EMA_26'] = ta.trend.EMAIndicator(close=df['Close'], window=26).ema_indicator()

    # 3. Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()

    # 4. MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()

    # 5. Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    df['Bollinger_Middle'] = bollinger.bollinger_mavg()

    # 6. Average True Range (ATR)
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()

    # 7. On-Balance Volume (OBV)
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['TickVol']).on_balance_volume()

    # 8. Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(high=df['High'], low=df['Low'], window1=9, window2=26, window3=52)
    df['Ichimoku_A'] = ichimoku.ichimoku_a()
    df['Ichimoku_B'] = ichimoku.ichimoku_b()

    # 9. Commodity Channel Index (CCI)
    df['CCI'] = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20).cci()

    # 10. Volume-Weighted Average Price (VWAP)
    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['TickVol']).volume_weighted_average_price()

    return df