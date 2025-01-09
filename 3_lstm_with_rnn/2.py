import logging
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
# import optuna  # <- REMOVED/COMMENTED OUT for single-run
import pandas as pd
import ta
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, SimpleRNN
from tensorflow.keras.optimizers import Adam

# 1) Suppress TensorFlow info messages (still show warnings/errors).
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 2) Optionally, ignore FutureWarning if anything else arises in the future
warnings.simplefilter(action='ignore', category=FutureWarning)

# Create directories if not exists
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/stock_prediction.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)  # Show all rows (optional, be cautious with large data)


def log_and_print(message):
    """Log and print the message."""
    print(message)
    logging.info(message)


# Fetch stock market data
def fetch_data(ticker, start_date, end_date):
    log_and_print(f"Fetching data for {ticker} from {start_date} to {end_date}")
    return yf.download(ticker, start=start_date, end=end_date)


# Add technical indicators
def add_technical_indicators(data):
    log_and_print("Adding technical indicators...")
    close_prices = data['Close'].squeeze()
    data['RSI'] = ta.momentum.RSIIndicator(close_prices, window=14).rsi()
    bol = ta.volatility.BollingerBands(close_prices, window=20)
    data['Lower_Band'] = bol.bollinger_lband()
    data['Upper_Band'] = bol.bollinger_hband()
    data['Daily_Log_Return'] = ta.others.DailyLogReturnIndicator(close_prices).daily_log_return()
    log_and_print("Technical indicators added successfully.")
    return data


# Prepare time-series data
def create_dataset(dataset, look_back=60):
    """
    dataset: (num_samples, num_features) after scaling
    look_back: how many timesteps to include in each sample
    """
    log_and_print("Creating time-series dataset...")
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i: i + look_back])
        y.append(dataset[i + look_back, 0])  # Predict 'Close' which is in column 0
    log_and_print("Time-series dataset created successfully.")
    return np.array(X), np.array(y)


def build_and_train_hybrid_model(params, X_train, y_train):
    log_and_print(f"Building hybrid RNN-LSTM model with params: {params}")

    # Add additional metrics in compile to track them during training
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        SimpleRNN(units=params['rnn_units'], return_sequences=True),
        Dropout(params['dropout_rate']),
        Bidirectional(LSTM(units=params['lstm_units'], return_sequences=False)),
        Dropout(params['dropout_rate']),
        Dense(units=25, activation='relu'),
        Dense(units=1)
    ])

    # Include MAE, MAPE as metrics for additional insight
    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='mean_squared_error',
        metrics=['mae', 'mape']
    )

    # Train with validation split
    history = model.fit(
        X_train, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_split=0.2,
        verbose=0
    )

    return model, history


def simulate_investments(
        model,
        X_test,
        y_test,
        data,
        scaler,
        look_back=60,
        initial_capital=1_000_000,
        transaction_amount=10_000
):
    """
    Simulate a simple trading strategy using the trained model predictions:
      - Start with `initial_capital` dollars in cash, 0 shares.
      - If the model predicts next day's price will be higher, buy `transaction_amount` worth of shares (if enough cash).
      - If the model predicts next day's price will be lower, sell `transaction_amount` worth of shares (if enough shares).
    """
    log_and_print("Starting investment simulation...")

    # 1) Generate predictions (scaled)
    scaled_preds = model.predict(X_test)

    # 2) Unscale predictions and actuals
    predictions_unscaled = scaler.inverse_transform(
        np.hstack((scaled_preds, np.zeros(
            (scaled_preds.shape[0], data[['RSI', 'Lower_Band', 'Upper_Band', 'Daily_Log_Return']].shape[1]))))
    )[:, 0]

    y_test_unscaled = scaler.inverse_transform(
        np.hstack((y_test.reshape(-1, 1),
                   np.zeros((y_test.shape[0], data[['RSI', 'Lower_Band', 'Upper_Band', 'Daily_Log_Return']].shape[1]))))
    )[:, 0]

    # 3) Identify the corresponding date indices for X_test
    test_start_idx = len(data) - len(X_test)
    test_dates = data.index[test_start_idx + look_back:]

    # 4) Initialize simulation variables
    cash = initial_capital
    shares_held = 0

    # We'll track daily portfolio values
    portfolio_values = []

    # 5) Loop through each test day
    for i in range(len(predictions_unscaled) - 1):
        today_date = test_dates[i]
        tomorrow_date = test_dates[i + 1]

        predicted_price_today = predictions_unscaled[i]
        predicted_price_tomorrow = predictions_unscaled[i + 1]
        actual_price_tomorrow = y_test_unscaled[i + 1]

        actual_price_today = y_test_unscaled[i]
        portfolio_value_today = cash + shares_held * actual_price_today

        if predicted_price_tomorrow > predicted_price_today:
            # Predicted to go up => BUY
            if cash >= transaction_amount:
                shares_to_buy = transaction_amount / actual_price_today
                cash -= transaction_amount
                shares_held += shares_to_buy

                log_and_print(
                    f"{today_date} - PREDICTING RISE for {tomorrow_date}: "
                    f"Buy ${transaction_amount} worth of shares at {actual_price_today:.2f}. "
                    f"New cash: {cash:.2f}, New shares held: {shares_held:.4f}"
                )
            else:
                log_and_print(f"{today_date} - PREDICTING RISE but NOT ENOUGH CASH to buy. Holding...")
        else:
            # Predicted to go down => SELL
            shares_needed_to_sell = transaction_amount / actual_price_today
            if shares_held >= shares_needed_to_sell:
                shares_held -= shares_needed_to_sell
                cash += transaction_amount
                log_and_print(
                    f"{today_date} - PREDICTING FALL for {tomorrow_date}: "
                    f"Sell ${transaction_amount} worth of shares at {actual_price_today:.2f}. "
                    f"New cash: {cash:.2f}, New shares held: {shares_held:.4f}"
                )
            else:
                log_and_print(f"{today_date} - PREDICTING FALL but NOT ENOUGH SHARES to sell. Holding...")

        log_and_print(
            f"  Predicted next-day price: {predicted_price_tomorrow:.2f} | "
            f"Actual next-day price: {actual_price_tomorrow:.2f}"
        )

        portfolio_values.append(portfolio_value_today)

    final_portfolio_value = cash + shares_held * y_test_unscaled[-1]
    log_and_print(f"Final Simulation Results:\n"
                  f"  Final Cash: {cash:.2f}\n"
                  f"  Final Shares Held: {shares_held:.4f}\n"
                  f"  Final Stock Price: {y_test_unscaled[-1]:.2f}\n"
                  f"  Final Portfolio Value: {final_portfolio_value:.2f}")

    return portfolio_values, final_portfolio_value


# ===================== MAIN SCRIPT STARTS HERE =====================

# Fetch data
ticker = "^GSPC"
start_date = "2020-01-01"
end_date = "2025-01-04"
data = fetch_data(ticker, start_date, end_date)

# Add technical indicators
data = add_technical_indicators(data)

# Handle missing values introduced by technical indicators
log_and_print("Handling missing values...")
data.ffill(inplace=True)
data.bfill(inplace=True)

# Check for any remaining nulls
if data.isnull().sum().sum() == 0:
    log_and_print("All missing values handled successfully.")
else:
    log_and_print("Warning: Missing values remain in the dataset!")

# Save cleaned dataset
csv_file_path = "cleaned_data_with_technical_indicators.csv"
data.to_csv(csv_file_path, index=True)
log_and_print(f"Cleaned dataset saved to CSV: {csv_file_path}")

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(
    data[['Close', 'RSI', 'Lower_Band', 'Upper_Band', 'Daily_Log_Return']].values
)
joblib.dump(scaler, "models/scaler.pkl")  # Save the scaler

# Create dataset with time-window
look_back = 60
X, y = create_dataset(scaled_data, look_back)

# ---------------- SPLIT into Train & Test Sets ----------------
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for Keras (batch_size, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# ------------------- SINGLE HYPERPARAMETER SET -------------------
# Removed Optuna optimization; using fixed parameters instead.
fixed_params = {
    "rnn_units": 50,
    "lstm_units": 50,
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50
}

# Train the model with the fixed hyperparameters
model, history = build_and_train_hybrid_model(fixed_params, X_train, y_train)

# OPTIONAL: You can save training/validation curves if needed
plt.figure(figsize=(14, 7))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title("Loss Over Epochs (Fixed Hyperparameters)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("plots/fixed_model_loss.png")
plt.close()

# Test set predictions for visualization
predictions = model.predict(X_test)
predictions_unscaled = scaler.inverse_transform(
    np.hstack((predictions, np.zeros((predictions.shape[0], scaled_data.shape[1] - 1))))
)[:, 0]
y_test_unscaled = scaler.inverse_transform(
    np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1))))
)[:, 0]

plt.figure(figsize=(14, 7))
plt.plot(y_test_unscaled, label="True Prices", color="blue")
plt.plot(predictions_unscaled, label="Predicted Prices", color="red")
plt.title("Predictions vs True Prices (Fixed Hyperparameters)")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.savefig("plots/fixed_model_predictions_vs_actual.png")
plt.close()

# Evaluate model performance on test set
test_mse = mean_squared_error(y_test_unscaled, predictions_unscaled)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test_unscaled, predictions_unscaled)
test_r2 = r2_score(y_test_unscaled, predictions_unscaled)

log_and_print(f"Fixed Model Test MSE: {test_mse}")
log_and_print(f"Fixed Model Test RMSE: {test_rmse}")
log_and_print(f"Fixed Model Test MAE: {test_mae}")
log_and_print(f"Fixed Model Test R2: {test_r2}")

# Simulate investments
portfolio_values, final_value = simulate_investments(
    model=model,
    X_test=X_test,
    y_test=y_test,
    data=data,
    scaler=scaler,
    look_back=look_back,
    initial_capital=1_000_000,
    transaction_amount=10_000
)

# Save the single-run model
fixed_model_folder = "models/fixed_params_model"
os.makedirs(fixed_model_folder, exist_ok=True)
model.save(f"{fixed_model_folder}/fixed_model.keras")
log_and_print(f"Fixed model saved to '{fixed_model_folder}/fixed_model.keras'")
