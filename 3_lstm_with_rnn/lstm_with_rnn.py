import logging
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ta
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score
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

# Replace fillna(method='ffill' / 'bfill') with direct ffill() / bfill()
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
# For time series, we typically take the last part for testing:
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for Keras (batch_size, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Define hyperparameter grid
param_grid = {
    'rnn_units': [50],
    'lstm_units': [50],
    'dropout_rate': [0.2],
    'learning_rate': [0.001],
    'batch_size': [32],
    'epochs': [50]
}

best_model = None
best_score = float('inf')
best_param_str = None  # We'll store the best model's param string here

# Hyperparameter optimization
from sklearn.model_selection import ParameterGrid

for params in ParameterGrid(param_grid):
    model, history = build_and_train_hybrid_model(params, X_train, y_train)

    # Evaluate final epoch's validation loss
    val_loss = history.history['val_loss'][-1]

    # Evaluate on test set for more robust comparison
    test_loss, test_mae, test_mape = model.evaluate(X_test, y_test, verbose=0)

    # Predict on test for additional stats
    test_preds = model.predict(X_test)
    test_preds_unscaled = scaler.inverse_transform(
        np.hstack((test_preds, np.zeros((test_preds.shape[0], 4))))
    )
    y_test_unscaled = scaler.inverse_transform(
        np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 4))))
    )

    # Additional metrics
    test_mse = mean_squared_error(y_test_unscaled[:, 0], test_preds_unscaled[:, 0])
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test_unscaled[:, 0], test_preds_unscaled[:, 0])

    # Create folders for the current param combination
    param_str = (
        f"units{params['lstm_units']}_dropout{params['dropout_rate']}"
        f"_lr{params['learning_rate']}_batch{params['batch_size']}_epochs{params['epochs']}"
    )
    model_folder = f"models/{param_str}"
    plot_folder = f"plots/{param_str}"
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    # Save the model
    model.save(f"{model_folder}/model.keras")

    # Plot training & validation loss
    plt.figure(figsize=(14, 7))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title(f"Loss for {param_str}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{plot_folder}/loss.png")
    plt.close()

    # Plot predictions vs. true (on the test set)
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_unscaled[:, 0], label="True Test Prices", color="blue")
    plt.plot(test_preds_unscaled[:, 0], label="Predicted Test Prices", color="red")
    plt.title(f"Test Predictions vs True Prices for {param_str}")
    plt.xlabel("Time Steps (Test Set)")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.savefig(f"{plot_folder}/test_predictions.png")
    plt.close()

    # Write stats to file
    with open(f"{model_folder}/stats.txt", "w") as stats_file:
        stats_file.write(f"Validation Loss: {val_loss}\n")
        stats_file.write(f"Test Loss (MSE): {test_loss}\n")
        stats_file.write(f"Test MAE: {test_mae}\n")
        stats_file.write(f"Test MAPE: {test_mape}\n")
        stats_file.write(f"Test MSE: {test_mse}\n")
        stats_file.write(f"Test RMSE: {test_rmse}\n")
        stats_file.write(f"Test R2 Score: {test_r2}\n")

    log_and_print(
        f"Params: {param_str} | "
        f"Val Loss: {val_loss:.5f} | "
        f"Test RMSE: {test_rmse:.5f} | Test R2: {test_r2:.5f}"
    )

    # Update best model based on val_loss (you might choose test_loss instead)
    if val_loss < best_score:
        best_score = val_loss
        best_model = model
        best_param_str = param_str

# Finally, log a short name rather than printing the entire config
log_and_print(f"Best model name: {best_param_str}")
log_and_print(f"Best score (Val Loss): {best_score}")

# todo: add preprocessing, such as removing outliers
