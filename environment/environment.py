import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ForexTradingEnv(gym.Env):
    """
    A custom OpenAI Gym environment for Forex trading that allows multiple positions and enforces specific closing rules,
    with margin and leverage considerations.
    """
    def __init__(self, df):
        super(ForexTradingEnv, self).__init__()

        self.long_positions_history = []
        self.short_positions_history = []

        self.df = df.reset_index(drop=True)  # Ensure the DataFrame index is sequential
        self.total_steps = len(self.df)

        # Initial account balance (Cent Account with $1,000 USD)
        self.initial_balance = 1000
        self.balance = self.initial_balance
        self.equity = self.initial_balance

        # Leverage and margin settings
        self.leverage = 200  # Leverage ratio
        self.margin_call_level = 100  # Margin Call Level (%)
        self.stop_out_level = 50  # Stop Out Level (%)

        # Margin variables
        self.used_margin = 0
        self.free_margin = self.balance - self.used_margin
        self.margin_level = np.inf  # Initially infinite when no positions are open

        # Positions will be stored in a list
        self.open_positions = []  # Each position is a dictionary with 'type', 'entry_price', and 'size'

        # For tracking history
        self.balance_history = []
        self.equity_history = []
        self.position_history = []
        self.used_margin_history = []
        self.free_margin_history = []
        self.margin_level_history = []

        # Exclude unscaled price columns from observation
        price_columns = ['Unscaled_Open', 'Unscaled_High', 'Unscaled_Low', 'Unscaled_Close', 'Spread', 'Unscaled_Spread']
        self.numeric_columns = [col for col in self.df.select_dtypes(include=[np.number]).columns.tolist() if col not in price_columns]

        # Define action and observation space
        # Action space: action_type
        # action_type: 0 - Hold, 1 - Buy, 2 - Sell, 3 - Close Positions
        self.action_space = spaces.Discrete(4)

        # Observation space: all numerical features except unscaled prices, plus margin variables
        num_features = len(self.numeric_columns) + 3  # Additional margin variables
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)

        # Initialize trade log
        self.trade_log = []

    def reset(self, *, seed=None, options=None):
        """
        Reset the state of the environment to an initial state.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.open_positions = []

        # Reset margin variables
        self.used_margin = 0
        self.free_margin = self.balance - self.used_margin
        self.margin_level = np.inf

        # Reset history
        self.balance_history = [self.balance]
        self.equity_history = [self.equity]
        self.position_history = [len(self.open_positions)]
        self.used_margin_history = [self.used_margin]
        self.free_margin_history = [self.free_margin]
        self.margin_level_history = [self.margin_level]

        # Initialize long and short positions history
        num_long_positions = 0
        num_short_positions = 0

        self.long_positions_history = [num_long_positions]
        self.short_positions_history = [num_short_positions]

        # Reset trade log
        self.trade_log = []

        return self._next_observation(), {}

    def _next_observation(self):
        """
        Get the observation for the current step.
        """
        # Extract the current row
        obs = self.df.loc[self.current_step, self.numeric_columns].values.astype(np.float32)

        # Replace any NaNs with zeros to ensure stability
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        # Add margin variables to observation
        # Handle infinite margin_level by capping it to a large finite value
        capped_margin_level = self.margin_level if np.isfinite(self.margin_level) else 1e6
        additional_obs = np.array([
            self.used_margin / self.initial_balance,
            self.free_margin / self.initial_balance,
            capped_margin_level / 1000  # To normalize
        ], dtype=np.float32)

        # Ensure no NaNs or Infs in additional_obs
        additional_obs = np.nan_to_num(additional_obs, nan=0.0, posinf=1e6 / 1000, neginf=0.0)

        # Concatenate the margin information to the observation
        obs = np.concatenate((obs, additional_obs))

        # Final check for NaNs in the observation
        if np.any(np.isnan(obs)):
            print(f"NaN detected in observation at step {self.current_step}")
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6 / 1000, neginf=0.0)

        return obs

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        terminated = False
        truncated = False

        # Store previous equity for reward calculation
        previous_equity = self.equity

        # Use unscaled close price for profit calculations
        current_price = self.df.loc[self.current_step, 'Unscaled_Close']
        current_spread = self.df.loc[self.current_step, 'Unscaled_Spread'] * 0.00001  # Convert spread to price units

        total_realized_profit = 0

        # Initialize penalty for invalid actions
        penalty = 0

        # Process action
        # action_type is an integer from 0 to 3
        action_type = action

        # Use fixed lot size
        lot_size = 0.1  # Fixed lot size
        position_size = lot_size * 100000  # Convert lot size to units

        if action_type == 0:
            # Hold, do nothing
            reward_penalty = -0.001  # Small penalty for holding
        elif action_type == 1 or action_type == 2:
            # Buy or Sell
            if len(self.open_positions) < 200:
                required_margin = position_size / self.leverage

                if self.free_margin >= required_margin and position_size > 0:
                    # Adjust entry price for spread
                    if action_type == 1:  # Buy
                        adjusted_entry_price = current_price + current_spread / 2
                        position_type = 'long'
                    else:  # Sell
                        adjusted_entry_price = current_price - current_spread / 2
                        position_type = 'short'

                    position = {
                        'type': position_type,
                        'entry_price': adjusted_entry_price,
                        'size': position_size
                    }
                    # Open position
                    self.open_positions.append(position)
                    # Update margins
                    self.used_margin += required_margin
                    self.free_margin -= required_margin
                    # Record the trade
                    trade = {
                        'step': self.current_step,
                        'date': self.df.loc[self.current_step, 'Date'],
                        'action': 'Buy' if action_type == 1 else 'Sell',
                        'position_size_lots': lot_size,
                        'entry_price': adjusted_entry_price
                    }
                    self.trade_log.append(trade)
                else:
                    # Not enough free margin or invalid position size
                    penalty -= 10  # Penalize invalid action
            else:
                # Too many open positions
                penalty -= 10  # Penalize invalid action
            reward_penalty = 0  # No penalty for taking action
        elif action_type == 3:
            # Attempt to close all positions
            positions_to_close = []
            for pos in self.open_positions:
                # Adjust exit price for spread
                if pos['type'] == 'long':
                    exit_price = current_price - current_spread / 2
                    pnl = (exit_price - pos['entry_price']) * pos['size']
                else:  # 'short'
                    exit_price = current_price + current_spread / 2
                    pnl = (pos['entry_price'] - exit_price) * pos['size']
                total_realized_profit += pnl
                # Update balance
                self.balance += pnl
                # Release margin
                required_margin = pos['size'] / self.leverage
                self.used_margin -= required_margin
                self.free_margin += required_margin
                positions_to_close.append(pos)
                # Record the trade
                trade = {
                    'step': self.current_step,
                    'date': self.df.loc[self.current_step, 'Date'],
                    'action': 'Close',
                    'position_type': 'Buy' if pos['type'] == 'long' else 'Sell',
                    'position_size_lots': pos['size'] / 100000,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'profit_loss': pnl
                }
                self.trade_log.append(trade)
            # Remove closed positions from open_positions
            for pos in positions_to_close:
                self.open_positions.remove(pos)
            reward_penalty = 0  # No penalty for taking action

        # Calculate unrealized PnL
        unrealized_pnl = 0
        for pos in self.open_positions:
            # Adjust potential exit price for spread
            if pos['type'] == 'long':
                potential_exit_price = current_price - current_spread / 2
                pnl = (potential_exit_price - pos['entry_price']) * pos['size']
            else:  # 'short'
                potential_exit_price = current_price + current_spread / 2
                pnl = (pos['entry_price'] - potential_exit_price) * pos['size']
            unrealized_pnl += pnl

        # Update equity
        self.equity = self.balance + unrealized_pnl

        # Update free margin
        self.free_margin = self.equity - self.used_margin

        # Update margin level, ensuring no division by zero
        if self.used_margin > 0:
            self.margin_level = (self.equity / self.used_margin) * 100
        else:
            self.margin_level = 1e6  # Assign a large finite value instead of infinity

        # Check for margin call / stop out
        if self.margin_level <= self.stop_out_level:
            # Close positions until margin level is above stop out level
            positions_to_close = []
            # Sort positions by PnL ascending (largest losses first)
            positions_with_pnl = []
            for pos in self.open_positions:
                # Adjust potential exit price for spread
                if pos['type'] == 'long':
                    exit_price = current_price - current_spread / 2
                    pnl = (exit_price - pos['entry_price']) * pos['size']
                else:  # 'short'
                    exit_price = current_price + current_spread / 2
                    pnl = (pos['entry_price'] - exit_price) * pos['size']
                positions_with_pnl.append({'position': pos, 'pnl': pnl})

            positions_with_pnl.sort(key=lambda x: x['pnl'])

            for p in positions_with_pnl:
                # Close position
                pnl = p['pnl']
                self.balance += pnl
                total_realized_profit += pnl

                # Release margin
                required_margin = p['position']['size'] / self.leverage
                self.used_margin -= required_margin
                self.free_margin += required_margin

                positions_to_close.append(p['position'])
                # Record the trade
                trade = {
                    'step': self.current_step,
                    'date': self.df.loc[self.current_step, 'Date'],
                    'action': 'Margin Call Close',
                    'position_type': 'Buy' if p['position']['type'] == 'long' else 'Sell',
                    'position_size_lots': p['position']['size'] / 100000,
                    'entry_price': p['position']['entry_price'],
                    'exit_price': exit_price,
                    'profit_loss': pnl
                }
                self.trade_log.append(trade)

                # Recalculate equity and margin level
                unrealized_pnl -= pnl  # Subtract the PnL of the closed position
                self.equity = self.balance + unrealized_pnl

                if self.used_margin > 0:
                    self.margin_level = (self.equity / self.used_margin) * 100
                else:
                    self.margin_level = 1e6  # Assign a large finite value instead of infinity

                if self.margin_level > self.stop_out_level:
                    break  # Margin level is acceptable

            # Remove closed positions from open_positions
            for pos in positions_to_close:
                if pos in self.open_positions:
                    self.open_positions.remove(pos)

        # Append history
        self.balance_history.append(self.balance)
        self.equity_history.append(self.equity)
        self.position_history.append(len(self.open_positions))
        self.used_margin_history.append(self.used_margin)
        self.free_margin_history.append(self.free_margin)
        self.margin_level_history.append(self.margin_level)

        num_long_positions = sum(1 for pos in self.open_positions if pos['type'] == 'long')
        num_short_positions = sum(1 for pos in self.open_positions if pos['type'] == 'short')

        self.long_positions_history.append(num_long_positions)
        self.short_positions_history.append(num_short_positions)

        # Move to the next step
        self.current_step += 1

        # Check if the episode is over
        if self.current_step >= self.total_steps - 1:
            terminated = True

        observation = self._next_observation()

        # Calculate reward based on equity change and drawdown penalty
        equity_change = (self.equity - previous_equity) / self.initial_balance

        # Calculate drawdown
        peak_equity = max(self.equity_history)
        drawdown = max(0, peak_equity - self.equity)
        drawdown_penalty_factor = 0.1  # Adjust as needed
        drawdown_penalty = (drawdown * drawdown_penalty_factor) / self.initial_balance

        # Calculate reward
        reward = equity_change - drawdown_penalty + penalty + reward_penalty

        # Final checks to prevent NaNs
        if not np.isfinite(self.equity):
            print(f"Non-finite equity detected at step {self.current_step}")
            self.equity = self.balance  # Reset equity to balance

        if not np.isfinite(reward):
            print(f"Non-finite reward detected at step {self.current_step}")
            reward = -1  # Assign a penalty

        return observation, reward, terminated, truncated, {}