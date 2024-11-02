from preprocessing.data_preprocessing import preprocess_data
from indicators.indicators import add_indicators
from environment.environment import ForexTradingEnv
from training.train_evaluate import train_model, evaluate_model
from visualization.visualization import display_trade_log, plot_trade_results
from stable_baselines3 import PPO
import os

# Step 1: Preprocess the data
df_train, df_test = preprocess_data('data/EURUSD-H1-16Years.csv')

# Step 2: Add technical indicators
df_train = add_indicators(df_train)
df_test = add_indicators(df_test)

# Step 3: Initialize the custom trading environment with the training data
env_train = ForexTradingEnv(df_train)

# Decide whether to train a new model or load an existing one
train_new_model = True  # Set to False to load an existing model

if train_new_model:
    # Step 4: Train the model using the best hyperparameters
    model = train_model(env_train)
else:
    # Step 4: Load the best model saved during hyperparameter optimization
    model_path = "training/models/best_model.zip"
    if os.path.exists(model_path):
        model = PPO.load(model_path)
    else:
        print("Best model not found. Please run hyperparameter_optimization.py first.")
        exit()

# Step 5: Evaluate the model with the test data
env_test = ForexTradingEnv(df_test)

trade_log = evaluate_model(model, env_test)

# Step 6: Display and plot the results
display_trade_log(trade_log)
plot_trade_results(env_test)