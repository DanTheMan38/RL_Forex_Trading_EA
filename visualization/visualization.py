import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def display_trade_log(trade_log):
    # Convert trade log to DataFrame
    trade_log_df = pd.DataFrame(trade_log)
    
    # Display settings for better readability
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Auto-adjust width based on the content
    pd.set_option('display.colheader_justify', 'left')  # Justify column headers to the left
    pd.set_option('display.float_format', '{:.5f}'.format)  # Limit float precision to 5 decimal places
    
    # Display the trade log
    print(trade_log_df)

def plot_trade_results(env_test):
    # Ensure 'Date' is a column in the DataFrame for plotting
    if 'Date' not in env_test.df.columns:
        env_test.df.reset_index(inplace=True)

    # Adjust dates to match the length of balance_history
    dates = env_test.df['Date'].iloc[:len(env_test.balance_history)].reset_index(drop=True)

    # 1. Plot Balance and Equity Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(dates, env_test.balance_history, label="Balance")
    plt.plot(dates, env_test.equity_history, label="Equity", color='green')
    plt.title("Balance and Equity Over Time")
    plt.xlabel("Date")
    plt.ylabel("Amount (USD)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Plot Profit Over Time
    plt.figure(figsize=(12, 6))
    profit_history = np.array(env_test.equity_history) - env_test.initial_balance
    plt.plot(dates, profit_history, label="Profit", color='orange')
    plt.title("Profit Over Time")
    plt.xlabel("Date")
    plt.ylabel("Profit (USD)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Plot Number of Open Positions Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(dates, env_test.position_history, label="Number of Open Positions")
    plt.title("Number of Open Positions Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Positions")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4. Plot Long and Short Positions Over Time
    if hasattr(env_test, 'long_positions_history') and hasattr(env_test, 'short_positions_history'):
        plt.figure(figsize=(12, 6))
        plt.plot(dates, env_test.long_positions_history, label="Number of Long Positions", color='blue')
        plt.plot(dates, env_test.short_positions_history, label="Number of Short Positions", color='red')
        plt.title("Long and Short Positions Over Time")
        plt.xlabel("Date")
        plt.ylabel("Number of Positions")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 6. Plot Used Margin and Free Margin Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(dates, env_test.used_margin_history, label="Used Margin", color='red')
    plt.plot(dates, env_test.free_margin_history, label="Free Margin", color='green')
    plt.title("Used Margin and Free Margin Over Time")
    plt.xlabel("Date")
    plt.ylabel("Amount (USD)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 7. Plot Drawdown Over Time
    plt.figure(figsize=(12, 6))
    equity_array = np.array(env_test.equity_history)
    peak_equity = np.maximum.accumulate(equity_array)
    drawdown = (peak_equity - equity_array) / peak_equity * 100
    plt.plot(dates, drawdown, label="Drawdown (%)", color='black')
    plt.title("Drawdown Over Time")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()