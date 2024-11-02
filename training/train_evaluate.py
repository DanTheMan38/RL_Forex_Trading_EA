from stable_baselines3 import PPO
from environment.environment import ForexTradingEnv
import numpy as np
import json

def train_model(env_train, total_timesteps=101326, model_name="best_model"):
    # Load best hyperparameters if they exist
    try:
        with open('best_hyperparameters.json', 'r') as f:
            best_params = json.load(f)
        print("Loaded best hyperparameters:", best_params)
    except FileNotFoundError:
        print("best_hyperparameters.json not found. Using default hyperparameters.")
        best_params = {}

    # Define the PPO model with the training environment
    model = PPO("MlpPolicy", env_train, verbose=1, **best_params)
    
    # Train the model
    model.learn(total_timesteps=total_timesteps)
    
    # Define the path to save the model
    model_path = "training/models/" + model_name

    # Save the trained model
    model.save(model_path)
    
    return model

def evaluate_model(model, env_test):
    # Reset the environment to start a new evaluation episode
    state, _ = env_test.reset()
    done = False
    truncated = False  # Initialize the truncated flag
    trade_log = []  # Initialize an empty list to store trade log information

    while not done and not truncated:
        # Get the action from your trained model
        action, _ = model.predict(state)
        
        # Extract action_type and lot_size from action array
        action_type = int(np.clip(action[0], 0, 3))
        lot_size = action[1]
        
        # Step the environment with the chosen action
        state, reward, done, truncated, info = env_test.step(action)
        
        # Only log trade actions (buy/sell), not hold
        if action_type in [1, 2]:  # Adjust based on your action mappings (1: Buy, 2: Sell)
            # Log relevant trade details
            trade_log.append({
                "action": "Buy" if action_type == 1 else "Sell",
                "lot_size": lot_size,
                "reward": reward,
                "equity": env_test.equity,
                "balance": env_test.balance,
                "free_margin": env_test.free_margin,
                "margin_level": env_test.margin_level,
                "num_open_positions": len(env_test.open_positions)
            })

    return trade_log