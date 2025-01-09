import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ta
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.optimizers import Adam

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
    log_and_print("Creating time-series dataset...")
    X, y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back)])
        y.append(dataset[i + look_back, 0])  # Predict 'Close'
    log_and_print("Time-series dataset created successfully.")
    return np.array(X), np.array(y)


def build_and_train_hybrid_model(params, X_train, y_train):
    log_and_print(f"Building hybrid RNN-LSTM model with params: {params}")

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        SimpleRNN(units=params['rnn_units'], return_sequences=True),
        Dropout(params['dropout_rate']),
        Bidirectional(LSTM(units=params['lstm_units'], return_sequences=False)),
        Dropout(params['dropout_rate']),
        Dense(units=25, activation='relu'),
        Dense(units=1)
    ])

    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mean_squared_error')

    history = model.fit(
        X_train, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_split=0.2,
        verbose=0
    )

    return model, history


# Fetch data for training
ticker = "^GSPC"
start_date = "2020-01-01"
end_date = "2025-01-04"
data = fetch_data(ticker, start_date, end_date)

# Add technical indicators
data = add_technical_indicators(data)

# Handle missing values introduced by technical indicators
log_and_print("Handling missing values...")

# Option 1: Fill missing values (forward fill, then backward fill)
data.fillna(method='ffill', inplace=True)  # Forward fill
data.fillna(method='bfill', inplace=True)  # Backward fill

# Option 2: Drop rows with missing values (e.g., first few rows from rolling indicators)
# Uncomment this if you prefer dropping instead of filling
# data.dropna(inplace=True)

# Ensure no missing values remain
if data.isnull().sum().sum() == 0:
    log_and_print("All missing values handled successfully.")
else:
    log_and_print("Warning: Missing values remain in the dataset!")

# Save the cleaned dataset for inspection
csv_file_path = "cleaned_data_with_technical_indicators.csv"
data.to_csv(csv_file_path, index=True)
log_and_print(f"Cleaned dataset saved to CSV: {csv_file_path}")

# Continue with further preprocessing and model preparation...


# Prepare training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close', 'RSI', 'Lower_Band', 'Upper_Band', 'Daily_Log_Return']].values)
joblib.dump(scaler, "models/scaler.pkl")  # Save scaler for later use

look_back = 60
X_train, y_train = create_dataset(scaled_data, look_back)

# Reshape data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

# Hyperparameter optimization
param_grid = {
    'rnn_units': [50, 100],
    'lstm_units': [50, 100],
    'dropout_rate': [0.2, 0.3],
    'learning_rate': [0.001, 0.0005],
    'batch_size': [32, 64],
    'epochs': [50, 100]
}

best_model = None
best_score = float('inf')

for params in ParameterGrid(param_grid):
    model, history = build_and_train_hybrid_model(params, X_train, y_train)

    # Create a separate folder for each parameter combination
    param_str = f"units{params['lstm_units']}_dropout{params['dropout_rate']}_lr{params['learning_rate']}_batch{params['batch_size']}_epochs{params['epochs']}"
    model_folder = f"models/{param_str}"
    plot_folder = f"plots/{param_str}"
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    # Save the model
    model.save(f"{model_folder}/model.keras")

    # Plot and save training/validation loss
    plt.figure(figsize=(14, 7))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title(f"Loss for {param_str}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{plot_folder}/loss.png")
    plt.close()

    # Evaluate on training data (as proxy for now)
    train_loss = history.history['val_loss'][-1]

    # Predict on training data for visualization
    predictions = model.predict(X_train)
    predicted_prices = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], 4)))))
    true_prices = scaler.inverse_transform(np.hstack((y_train.reshape(-1, 1), np.zeros((y_train.shape[0], 4)))))

    # Save prediction plot
    plt.figure(figsize=(14, 7))
    plt.plot(true_prices[:, 0], label="True Prices", color="blue")
    plt.plot(predicted_prices[:, 0], label="Predicted Prices", color="red")
    plt.title(f"Predictions vs True Prices for {param_str}")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.savefig(f"{plot_folder}/predictions.png")
    plt.close()

    # Save statistics (e.g., loss, RMSE, R2) to a log file
    mse = mean_squared_error(true_prices[:, 0], predicted_prices[:, 0])
    rmse = np.sqrt(mse)
    r2 = r2_score(true_prices[:, 0], predicted_prices[:, 0])

    with open(f"{model_folder}/stats.txt", "w") as stats_file:
        stats_file.write(f"MSE: {mse}\n")
        stats_file.write(f"RMSE: {rmse}\n")
        stats_file.write(f"R2 Score: {r2}\n")
        stats_file.write(f"Validation Loss: {train_loss}\n")

    if train_loss < best_score:
        best_model = model
        best_score = train_loss

log_and_print(f"Best model parameters: {best_model.get_config()}")
log_and_print(f"Best score: {best_score}")

# todo: add preprocessing, such as removing outliers
