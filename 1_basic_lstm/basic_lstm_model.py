import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

matplotlib.use("TkAgg")  # Use the TkAgg backend for plotting

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

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


# Prepare time-series data
def create_dataset(dataset, look_back=60):
    log_and_print("Creating time-series dataset...")
    X, y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        y.append(dataset[i + look_back, 0])
    log_and_print("Time-series dataset created successfully.")
    return np.array(X), np.array(y)


# Fetch data for training
ticker = "^GSPC"
start_date = "2015-01-01"
end_date = "2023-12-31"
data = fetch_data(ticker, start_date, end_date)

# Prepare training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']].values)
look_back = 60
X_train, y_train = create_dataset(scaled_data, look_back)

# Reshape data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
log_and_print("Building the LSTM model...")
model = Sequential([
    Input(shape=(X_train.shape[1], 1)),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])
log_and_print("Model built successfully.")

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Add TensorBoard callback
tensorboard_callback = TensorBoard(log_dir="logs/tensorboard", histogram_freq=1)

# Train the model
log_and_print("Starting model training...")
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[tensorboard_callback])
log_and_print("Model training completed.")

# Fetch test data
test_start_date = "2020-01-01"
test_end_date = "2025-01-04"
test_data = fetch_data(ticker, test_start_date, test_end_date)

scaled_test_data = scaler.transform(test_data[['Close']].values)
X_test, y_test = create_dataset(scaled_test_data, look_back)

# Reshape for prediction
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict
log_and_print("Starting predictions...")
predictions = model.predict(X_test)
log_and_print("Predictions completed.")

# Inverse scale predictions
predicted_prices = scaler.inverse_transform(predictions)
true_prices = scaler.inverse_transform([y_test])

# Metrics
mse = mean_squared_error(true_prices[0], predicted_prices[:, 0])
rmse = np.sqrt(mse)
r2 = r2_score(true_prices[0], predicted_prices[:, 0])
mae = mean_absolute_error(true_prices[0], predicted_prices[:, 0])
explained_variance = explained_variance_score(true_prices[0], predicted_prices[:, 0])
direction_accuracy = np.mean((np.sign(true_prices[0][1:] - true_prices[0][:-1]) ==
                              np.sign(predicted_prices[1:, 0] - predicted_prices[:-1, 0])).astype(int))

log_and_print(f"Mean Squared Error: {mse}")
log_and_print(f"Root Mean Squared Error: {rmse}")
log_and_print(f"R^2 Score: {r2}")
log_and_print(f"Mean Absolute Error: {mae}")
log_and_print(f"Explained Variance Score: {explained_variance}")
log_and_print(f"Directional Accuracy: {direction_accuracy * 100:.2f}%")

# Save model summary
with open("logs/model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
log_and_print("Model summary saved.")


# Plot training and validation loss
plt.figure(figsize=(14, 7))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title("Model Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("plots/loss_plot.png")
log_and_print("Training and validation loss plot saved.")

# Plot predictions vs. true prices
plt.figure(figsize=(14, 7))
plt.plot(test_data.index[look_back + 1:], true_prices[0], label="True Prices", color="blue")
plt.plot(test_data.index[look_back + 1:], predicted_prices[:, 0], label="Predicted Prices", color="red")
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.savefig("plots/prediction_plot.png")
log_and_print("Prediction plot saved successfully.")

# Residual analysis
residuals = true_prices[0] - predicted_prices[:, 0]
plt.figure(figsize=(14, 7))
plt.plot(residuals, label='Residuals', color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.title("Residual Analysis")
plt.xlabel("Time")
plt.ylabel("Residuals")
plt.legend()
plt.savefig("plots/residuals_plot.png")
log_and_print("Residual analysis plot saved.")

# Rolling RMSE
window_size = 50
rolling_rmse = [np.sqrt(np.mean((residuals[i:i + window_size]) ** 2)) for i in range(len(residuals) - window_size)]
plt.figure(figsize=(14, 7))
plt.plot(rolling_rmse, label='Rolling RMSE', color='green')
plt.title("Rolling RMSE Over Time")
plt.xlabel("Time")
plt.ylabel("RMSE")
plt.legend()
plt.savefig("plots/rolling_rmse.png")
log_and_print("Rolling RMSE plot saved.")

# Prediction error distribution
plt.figure(figsize=(14, 7))
plt.hist(residuals, bins=50, color='blue', alpha=0.7)
plt.title("Prediction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.savefig("plots/error_distribution.png")
log_and_print("Error distribution plot saved.")
