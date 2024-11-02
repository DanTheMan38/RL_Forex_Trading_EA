import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from preprocessing.data_preprocessing import preprocess_data
from indicators.indicators import add_indicators
from environment.environment import ForexTradingEnv
import json

def objective(trial):
    # Define hyperparameter search space
    n_steps = trial.suggest_categorical('n_steps', [128, 256, 512, 1024])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.5, 1.0)
    vf_coef = trial.suggest_float('vf_coef', 0.5, 1.0)

    # Preprocess data
    df_train, df_test = preprocess_data('data/EURUSD-H1-16Years.csv')
    df_train = add_indicators(df_train)
    df_test = add_indicators(df_test)

    # Initialize environment
    env = ForexTradingEnv(df_train)

    # Define the model
    model = PPO(
        'MlpPolicy',
        env,
        verbose=0,
        tensorboard_log='logs/',
        n_steps=n_steps,
        gamma=gamma,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        vf_coef=vf_coef,
    )

    # Evaluate the model using EvalCallback
    eval_env = Monitor(ForexTradingEnv(df_test))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='training/models',
        log_path='logs/',
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # Train the model
    model.learn(total_timesteps=100000, callback=eval_callback)

    # Retrieve evaluation results
    mean_reward = eval_callback.best_mean_reward

    return -mean_reward  # Optuna minimizes the objective

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)

    # Save the best hyperparameters
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(study.best_params, f)
    
    print("Best model saved to 'training/models/best_model.zip'")