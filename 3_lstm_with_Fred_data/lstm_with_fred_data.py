import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ta
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

matplotlib.use("TkAgg")  # Use the TkAgg backend for plotting

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/stock_prediction.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


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


# Fetch data for training
ticker = "^GSPC"
start_date = "2015-01-01"
end_date = "2023-12-31"
data = fetch_data(ticker, start_date, end_date)

# Add technical indicators
data = add_technical_indicators(data)

# Handle missing values introduced by indicators
data = data.dropna()

# Prepare training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close', 'RSI', 'Lower_Band', 'Upper_Band', 'Daily_Log_Return']].values)

look_back = 60
X_train, y_train = create_dataset(scaled_data, look_back)

# Reshape data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

# Build the LSTM model
log_and_print("Building the LSTM model...")
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),  # Explicit input layer
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),  # Regularization
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])
log_and_print("Model built successfully.")

# Compile the model with Adam optimizer and learning rate tuning
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model with validation split
log_and_print("Starting model training...")
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)
log_and_print("Model training completed.")

# Fetch test data
test_start_date = "2024-01-01"
test_end_date = "2024-12-31"
test_data = fetch_data(ticker, test_start_date, test_end_date)

# Add technical indicators to test data
test_data = add_technical_indicators(test_data)

# Handle missing values in test data
test_data = test_data.dropna()

scaled_test_data = scaler.transform(test_data[['Close', 'RSI', 'Lower_Band', 'Upper_Band', 'Daily_Log_Return']].values)
X_test, y_test = create_dataset(scaled_test_data, look_back)

# Reshape for prediction
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Predict
log_and_print("Starting predictions...")
predictions = model.predict(X_test)
log_and_print("Predictions completed.")

# Inverse scale predictions
predicted_prices = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], 4)))))
true_prices = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 4)))))

# Metrics
mse = mean_squared_error(true_prices[:, 0], predicted_prices[:, 0])
rmse = np.sqrt(mse)
r2 = r2_score(true_prices[:, 0], predicted_prices[:, 0])

log_and_print(f"Mean Squared Error: {mse}")
log_and_print(f"Root Mean Squared Error: {rmse}")
log_and_print(f"R^2 Score: {r2}")

# Plot training and validation loss
plt.figure(figsize=(14, 7))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title("Model Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("plots/loss_plot.png")  # Save loss plot

# Plot predictions vs. true prices
plt.figure(figsize=(14, 7))
plt.plot(test_data.index[look_back + 1:], true_prices[:, 0], label="True Prices", color="blue")
plt.plot(test_data.index[look_back + 1:], predicted_prices[:, 0], label="Predicted Prices", color="red")
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.savefig("plots/prediction_plot.png")  # Save prediction plot
log_and_print("Plots saved successfully.")
