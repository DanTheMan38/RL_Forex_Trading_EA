# RL Forex Trading EA

A Forex trading expert advisor (EA) powered by reinforcement learning for automated trading.

## Overview

This project is a custom OpenAI Gym environment for Forex trading that allows multiple positions and enforces specific closing rules, with margin and leverage considerations. The environment is designed to train a reinforcement learning agent to trade in the Forex market using historical data.

**Note**: This project is still under development (approximately 80% complete).

## Features

- Custom Forex trading environment compatible with OpenAI Gym API.
- Data preprocessing and feature engineering, including technical indicators.
- Training using Proximal Policy Optimization (PPO) from Stable Baselines3.
- Hyperparameter optimization using Optuna.
- Evaluation and visualization of trading performance.

## Project Structure

- `environment/`:
  - `environment.py`: Defines the `ForexTradingEnv` class.
- `preprocessing/`:
  - `data_preprocessing.py`: Functions for data loading and preprocessing.
- `indicators/`:
  - `indicators.py`: Functions to add technical indicators.
- `training/`:
  - `hyperparameter_optimization.py`: Hyperparameter tuning with Optuna.
  - `train_evaluate.py`: Training and evaluation scripts.
- `visualization/`:
  - `visualization.py`: Functions to display trade logs and plot results.
- `main.py`: Main script to run the training and evaluation pipeline.
- `data/`: Directory for storing data files.

## Installation

```markdown
# Clone the repository
git clone https://github.com/yourusername/RL_Forex_Trading_EA.git
cd RL_Forex_Trading_EA

# Install the required packages
pip install -r requirements.txt

## Usage

# Step 1: Prepare the Data
# Place your historical Forex data in the `data/` directory.
# Ensure the data is in the format expected by `data_preprocessing.py`.

# Step 2: Preprocess the Data
# Run the preprocessing script
python preprocessing/data_preprocessing.py

# Step 3: Add Technical Indicators
# Enhance the data with technical indicators
python indicators/indicators.py

# Step 4: Train the Model
# Run the training script
python training/train_evaluate.py

# Step 5: Hyperparameter Optimization (Optional)
# To perform hyperparameter tuning with Optuna, run
python training/hyperparameter_optimization.py

# Step 6: Evaluate the Model
# Evaluation is integrated into the training pipeline.

# Step 7: Visualize the Results
# Generate plots of the trading performance
python visualization/visualization.py

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- ta
- gymnasium
- stable-baselines3
- optuna
- matplotlib