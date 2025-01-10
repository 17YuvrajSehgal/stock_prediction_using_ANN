import logging
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna  # Re-enabled Optuna for hyperparameter optimization
import pandas as pd
import ta
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping  # Added for Early Stopping
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, SimpleRNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2  # Added for L2 regularization

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
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        log_and_print("No data fetched. Please check the ticker and date range.")
    return data


# Add technical indicators
def add_technical_indicators(data):
    log_and_print("Adding technical indicators...")
    close_prices = data['Close'].squeeze()
    data['RSI'] = ta.momentum.RSIIndicator(close_prices, window=14).rsi()
    bol = ta.volatility.BollingerBands(close_prices, window=20)
    data['Lower_Band'] = bol.bollinger_lband()
    data['Upper_Band'] = bol.bollinger_hband()
    data['Daily_Log_Return'] = ta.others.DailyLogReturnIndicator(close_prices).daily_log_return()

    # Added MACD as a new technical indicator
    macd = ta.trend.MACD(close_prices)
    data['MACD'] = macd.macd()

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
        # Added L2 regularization to SimpleRNN layer
        SimpleRNN(units=params['rnn_units'], return_sequences=True,
                  kernel_regularizer=l2(params['l2_reg'])),
        Dropout(params['dropout_rate']),
        # Added L2 regularization to LSTM layer
        Bidirectional(LSTM(units=params['lstm_units'], return_sequences=False,
                           kernel_regularizer=l2(params['l2_reg']))),
        Dropout(params['dropout_rate']),
        Dense(units=25, activation='relu', kernel_regularizer=l2(params['l2_reg'])),
        Dense(units=1)
    ])

    # Include MAE, MAPE as metrics for additional insight
    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='mean_squared_error',
        metrics=['mae', 'mape']
    )

    # Added EarlyStopping callback to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=2
    )

    # Train with validation split and EarlyStopping
    history = model.fit(
        X_train, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_split=0.2,
        verbose=2,
        callbacks=[early_stopping]  # Added callbacks parameter
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
        transaction_amount=50_000
):
    """
    Simulate a simple trading strategy using the trained model predictions:
      - Start with `initial_capital` dollars in cash, 0 shares.
      - If the model predicts next day's price will be higher, buy `transaction_amount` worth of shares (if enough cash).
      - If the model predicts next day's price will be lower, sell `transaction_amount` worth of shares (if enough shares).

    Args:
        model (keras.Model): Trained model.
        X_test (np.array): Feature data for test set (shape: [samples, timesteps, features]).
        y_test (np.array): True prices (scaled) for the test set.
        data (pd.DataFrame): Original dataframe with dates and close prices (for indexing).
        scaler (MinMaxScaler): Fitted scaler to invert predictions.
        look_back (int): Number of look-back steps used in creating the dataset.
        initial_capital (float): Starting cash.
        transaction_amount (float): Amount in $ to invest or withdraw each time the model signals a move.
    """
    log_and_print("Starting investment simulation...")

    # 1) Generate predictions (scaled)
    scaled_preds = model.predict(X_test)

    # 2) Unscale predictions and actuals
    # Adjusted to include the new 'MACD' feature by adding an additional zero column
    # Now, we have 6 features: ['Close', 'RSI', 'Lower_Band', 'Upper_Band', 'Daily_Log_Return', 'MACD']
    num_features = scaler.scale_.shape[0]
    scaled_preds_extended = np.hstack((scaled_preds, np.zeros((scaled_preds.shape[0], num_features - 1))))
    predictions_unscaled = scaler.inverse_transform(scaled_preds_extended)[:, 0]

    # Similarly adjust y_test
    y_test_extended = np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], num_features - 1))))
    y_test_unscaled = scaler.inverse_transform(y_test_extended)[:, 0]

    # 3) Identify the corresponding date indices for X_test
    test_start_idx = len(data) - len(X_test)
    # Attempt to include one extra date to prevent IndexError
    try:
        test_dates = data.index[test_start_idx + look_back: test_start_idx + look_back + len(X_test) + 1]
        if len(test_dates) < len(X_test) + 1:
            raise IndexError("Not enough dates for simulation.")
    except IndexError:
        log_and_print("Warning: Not enough dates in test_dates to simulate investments for all predictions.")
        # Adjust test_dates to have the necessary length by excluding the last prediction
        test_dates = data.index[test_start_idx + look_back: test_start_idx + look_back + len(X_test)]
        log_and_print("Adjusted test_dates to match available data. The last prediction will be skipped.")

    # 4) Initialize simulation variables
    cash = initial_capital
    shares_held = 0

    # We'll track daily portfolio values and trade dates
    portfolio_values = []
    trade_dates = []

    # 5) Check if the index is unique
    if not data.index.is_unique:
        log_and_print("Warning: Data index is not unique. Dropping duplicate dates, keeping first occurrence.")
        data = data[~data.index.duplicated(keep='first')]

    # 6) Loop through each test day safely using zip to prevent IndexError
    for today_date, tomorrow_date, pred_today, pred_tomorrow, actual_tomorrow in zip(
            test_dates[:-1],  # All but the last date
            test_dates[1:],  # All but the first date
            predictions_unscaled[:-1],
            predictions_unscaled[1:],
            y_test_unscaled[1:]
    ):
        # Ensure today_date is a scalar and matches the DataFrame's index type
        if isinstance(today_date, pd.Timestamp):
            pass
        elif isinstance(today_date, np.datetime64):
            today_date = pd.Timestamp(today_date)
        elif isinstance(today_date, str):
            today_date = pd.Timestamp(today_date)
        else:
            today_date = pd.Timestamp(today_date)

        # Retrieve actual_price_today safely
        try:
            actual_price_today = data.at[today_date, 'Close']
        except (KeyError, TypeError):
            # If multiple entries exist or type mismatch, select the first one
            actual_price_today = data.loc[today_date, 'Close']
            if isinstance(actual_price_today, pd.Series) or isinstance(actual_price_today, pd.DataFrame):
                actual_price_today = actual_price_today.iloc[0]
            elif isinstance(actual_price_today, np.ndarray):
                actual_price_today = actual_price_today[0]
            else:
                actual_price_today = float(actual_price_today)

        # Ensure actual_price_today is a float
        actual_price_today = float(actual_price_today)

        portfolio_value_today = cash + shares_held * actual_price_today

        # Determine action based on prediction for tomorrow vs today
        if pred_tomorrow > pred_today:
            # Predicted to go up => BUY
            if cash >= transaction_amount:
                shares_to_buy = transaction_amount / actual_price_today
                cash -= transaction_amount
                shares_held += shares_to_buy

                log_and_print(
                    f"{today_date.date()} - PREDICTING RISE for {tomorrow_date.date()}: "
                    f"Buy ${transaction_amount} worth of shares at ${actual_price_today:.2f}. "
                    f"New cash: ${cash:.2f}, New shares held: {shares_held:.4f}"
                )
            else:
                log_and_print(f"{today_date.date()} - PREDICTING RISE but NOT ENOUGH CASH to buy. Holding...")
        else:
            # Predicted to go down => SELL
            shares_needed_to_sell = transaction_amount / actual_price_today
            if shares_held >= shares_needed_to_sell:
                shares_held -= shares_needed_to_sell
                cash += transaction_amount
                log_and_print(
                    f"{today_date.date()} - PREDICTING FALL for {tomorrow_date.date()}: "
                    f"Sell ${transaction_amount} worth of shares at ${actual_price_today:.2f}. "
                    f"New cash: ${cash:.2f}, New shares held: {shares_held:.4f}"
                )
            else:
                log_and_print(f"{today_date.date()} - PREDICTING FALL but NOT ENOUGH SHARES to sell. Holding...")

        # Log predicted vs actual for tomorrow
        log_and_print(
            f"  Predicted next-day price: ${pred_tomorrow:.2f} | "
            f"Actual next-day price: ${actual_tomorrow:.2f}"
        )

        # Record portfolio value after today's action, valued at today's actual price
        portfolio_values.append(portfolio_value_today)
        trade_dates.append(today_date)

    # After loop ends, log final state
    # The last day’s portfolio value is valued at the last actual price
    final_portfolio_value = cash + shares_held * y_test_unscaled[-1]
    profit_loss = final_portfolio_value - initial_capital
    log_and_print(f"Final Simulation Results:\n"
                  f"  Final Cash: ${cash:.2f}\n"
                  f"  Final Shares Held: {shares_held:.4f}\n"
                  f"  Final Stock Price: ${y_test_unscaled[-1]:.2f}\n"
                  f"  Final Portfolio Value: ${final_portfolio_value:.2f}\n"
                  f"  Total Profit/Loss: ${profit_loss:.2f}")

    # Convert portfolio_values and trade_dates to pandas Series for easier computation
    portfolio_series = pd.Series(portfolio_values, index=trade_dates)

    # Calculate daily returns
    daily_returns = portfolio_series.pct_change().dropna()

    # Calculate Sharpe Ratio (Assuming 252 trading days in a year)
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0.0

    # Calculate Max Drawdown
    cumulative_max = portfolio_series.cummax()
    drawdowns = (portfolio_series - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()

    log_and_print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    log_and_print(f"Max Drawdown: {max_drawdown:.4f}")

    # Plot Equity Curve and Drawdown
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_series.index, portfolio_series.values, label='Portfolio Value', color='green')
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.savefig("plots/equity_curve.png")
    plt.close()

    plt.figure(figsize=(14, 7))
    plt.plot(drawdowns.index, drawdowns.values, label='Drawdown', color='red')
    plt.title("Drawdown Over Time")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.savefig("plots/drawdown.png")
    plt.close()

    # After loop ends, log final state with new metrics
    log_and_print(f"Final Simulation Results with Enhanced Metrics:\n"
                  f"  Final Cash: ${cash:.2f}\n"
                  f"  Final Shares Held: {shares_held:.4f}\n"
                  f"  Final Stock Price: ${y_test_unscaled[-1]:.2f}\n"
                  f"  Final Portfolio Value: ${final_portfolio_value:.2f}\n"
                  f"  Total Profit/Loss: ${profit_loss:.2f}\n"
                  f"  Sharpe Ratio: {sharpe_ratio:.4f}\n"
                  f"  Max Drawdown: {max_drawdown:.4f}")

    # Return details for further analysis if desired
    return portfolio_values, trade_dates, final_portfolio_value, profit_loss, sharpe_ratio, max_drawdown


# ===================== MAIN SCRIPT STARTS HERE =====================

if __name__ == "__main__":
    # Fetch data
    ticker = "^GSPC"
    start_date = "2020-01-01"
    end_date = "2025-01-09"
    data = fetch_data(ticker, start_date, end_date)

    if data.empty:
        log_and_print("No data fetched. Exiting the script.")
        exit(1)

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

    # Ensure the DataFrame index is unique to prevent access issues
    if not data.index.is_unique:
        log_and_print("Data index is not unique. Dropping duplicate dates, keeping first occurrence.")
        data = data[~data.index.duplicated(keep='first')]

    # Save cleaned dataset
    csv_file_path = "cleaned_data_with_technical_indicators.csv"
    data.to_csv(csv_file_path, index=True)
    log_and_print(f"Cleaned dataset saved to CSV: {csv_file_path}")

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Updated to include 'MACD' in the scaling
    scaled_data = scaler.fit_transform(
        data[['Close', 'RSI', 'Lower_Band', 'Upper_Band', 'Daily_Log_Return', 'MACD']].values
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


    # ------------------- OPTUNA HYPERPARAMETER OPTIMIZATION -------------------

    def objective(trial):
        # Suggest hyperparameters for this trial
        rnn_units = trial.suggest_int("rnn_units", 10, 100, step=10)
        lstm_units = trial.suggest_int("lstm_units", 10, 100, step=10)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs = trial.suggest_int("epochs", 10, 100, step=10)
        l2_reg = trial.suggest_float("l2_reg", 1e-5, 1e-2, log=True)  # Added L2 regularization hyperparameter

        # Build and train the model
        params = {
            "rnn_units": rnn_units,
            "lstm_units": lstm_units,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "l2_reg": l2_reg,  # Added to params
        }
        model, history = build_and_train_hybrid_model(params, X_train, y_train)

        # Evaluate validation loss (last epoch)
        val_loss = history.history['val_loss'][-1]

        # Create folders for this trial
        param_str = (
            f"units{params['lstm_units']}_dropout{params['dropout_rate']}"
            f"_lr{params['learning_rate']}_batch{params['batch_size']}_epochs{params['epochs']}"
            f"_l2{params['l2_reg']}"
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

        # Predictions vs. True Values on Test Set
        predictions = model.predict(X_test)
        # Adjusted to include 'MACD' by adding a zero column
        scaled_predictions_extended = np.hstack(
            (predictions, np.zeros((predictions.shape[0], scaled_data.shape[1] - 1))))
        predictions_unscaled = scaler.inverse_transform(scaled_predictions_extended)[:, 0]
        # Similarly adjust y_test
        y_test_extended_val = np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1))))
        y_test_unscaled_val = scaler.inverse_transform(y_test_extended_val)[:, 0]

        plt.figure(figsize=(14, 7))
        plt.plot(y_test_unscaled_val, label="True Prices", color="blue")
        plt.plot(predictions_unscaled, label="Predicted Prices", color="red")
        plt.title(f"Predictions vs True Prices for {param_str}")
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.legend()
        plt.savefig(f"{plot_folder}/predictions_vs_actual.png")
        plt.close()

        # ------------------- SIMULATE INVESTMENTS FOR THIS TRIAL -------------------
        portfolio_values, trade_dates, final_portfolio_value, profit_loss, sharpe_ratio, max_drawdown = simulate_investments(
            model=model,
            X_test=X_test,
            y_test=y_test,
            data=data,
            scaler=scaler,
            look_back=look_back,
            initial_capital=1_000_000,
            transaction_amount=50_000
        )

        # Plot Portfolio Values Over Time
        plt.figure(figsize=(14, 7))
        plt.plot(trade_dates, portfolio_values, label="Portfolio Value", color="green")
        plt.title(f"Portfolio Value Over Time for {param_str}")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.savefig(f"{plot_folder}/portfolio_values.png")
        plt.close()

        # Save stats
        with open(f"{model_folder}/stats.txt", "w") as stats_file:
            stats_file.write(f"Validation Loss: {val_loss}\n")
            test_mse = mean_squared_error(y_test_unscaled_val, predictions_unscaled)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test_unscaled_val, predictions_unscaled)
            test_r2 = r2_score(y_test_unscaled_val, predictions_unscaled)
            stats_file.write(f"Test MSE: {test_mse}\n")
            stats_file.write(f"Test RMSE: {test_rmse}\n")
            stats_file.write(f"Test MAE: {test_mae}\n")
            stats_file.write(f"Test R2 Score: {test_r2}\n")
            stats_file.write(f"Final Portfolio Value: ${final_portfolio_value:.2f}\n")
            stats_file.write(f"Total Profit/Loss: ${profit_loss:.2f}\n")
            stats_file.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
            stats_file.write(f"Max Drawdown: {max_drawdown:.4f}\n")

        # Report the validation loss to Optuna
        trial.report(val_loss, step=epochs)

        # If validation loss is worse than the pruning threshold, prune the trial
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return val_loss


    # Create Optuna study
    study = optuna.create_study(direction="minimize")

    # Optimize the study with the objective function
    study.optimize(objective, n_trials=50, timeout=3600)

    # Log the best hyperparameters and their corresponding score
    best_params = study.best_params
    best_val_loss = study.best_value

    log_and_print(f"Best Parameters: {best_params}")
    log_and_print(f"Best Validation Loss: {best_val_loss}")

    # ------------------- TRAIN THE BEST MODEL -------------------
    # Build and train the best model with the optimal hyperparameters
    best_model, best_history = build_and_train_hybrid_model(best_params, X_train, y_train)

    # Create a unique string for the best model's parameters
    best_param_str = (
        f"units{best_params['lstm_units']}_dropout{best_params['dropout_rate']}"
        f"_lr{best_params['learning_rate']}_batch{best_params['batch_size']}_epochs{best_params['epochs']}"
        f"_l2{best_params['l2_reg']}"
    )
    best_model_folder = f"models/{best_param_str}"
    plot_folder = f"plots/{best_param_str}"
    os.makedirs(best_model_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    # Save the best model
    best_model.save(f"{best_model_folder}/best_model.keras")
    log_and_print(f"Best model saved to '{best_model_folder}/best_model.keras'")

    # Plot training & validation loss for the best model
    plt.figure(figsize=(14, 7))
    plt.plot(best_history.history['loss'], label='Training Loss', color='blue')
    plt.plot(best_history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title("Loss Over Epochs (Best Hyperparameters)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{plot_folder}/best_model_loss.png")
    plt.close()

    # Test set predictions for visualization
    predictions = best_model.predict(X_test)
    # Adjusted to include 'MACD' by adding a zero column
    scaled_predictions_extended = np.hstack((predictions, np.zeros((predictions.shape[0], scaled_data.shape[1] - 1))))
    predictions_unscaled = scaler.inverse_transform(scaled_predictions_extended)[:, 0]
    # Similarly adjust y_test
    y_test_extended_final = np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1))))
    y_test_unscaled_final = scaler.inverse_transform(y_test_extended_final)[:, 0]

    plt.figure(figsize=(14, 7))
    plt.plot(y_test_unscaled_final, label="True Prices", color="blue")
    plt.plot(predictions_unscaled, label="Predicted Prices", color="red")
    plt.title("Predictions vs True Prices (Best Hyperparameters)")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(f"{plot_folder}/best_model_predictions_vs_actual.png")
    plt.close()

    # Evaluate model performance on test set
    test_mse = mean_squared_error(y_test_unscaled_final, predictions_unscaled)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test_unscaled_final, predictions_unscaled)
    test_r2 = r2_score(y_test_unscaled_final, predictions_unscaled)

    log_and_print(f"Best Model Test MSE: {test_mse}")
    log_and_print(f"Best Model Test RMSE: {test_rmse}")
    log_and_print(f"Best Model Test MAE: {test_mae}")
    log_and_print(f"Best Model Test R2: {test_r2}")

    # ------------------- SIMULATE INVESTMENTS FOR THE BEST MODEL -------------------
    portfolio_values, trade_dates, final_value, profit_loss, sharpe_ratio, max_drawdown = simulate_investments(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        data=data,
        scaler=scaler,
        look_back=look_back,
        initial_capital=1_000_000,
        transaction_amount=50_000
    )

    # Plot Portfolio Values Over Time for the Best Model
    plt.figure(figsize=(14, 7))
    plt.plot(trade_dates, portfolio_values, label="Portfolio Value", color="green")
    plt.title("Portfolio Value Over Time (Best Hyperparameters)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.savefig(f"{plot_folder}/best_model_portfolio_values.png")
    plt.close()

    # Plot Equity Curve and Drawdown for the Best Model
    portfolio_series_best = pd.Series(portfolio_values, index=trade_dates)
    daily_returns_best = portfolio_series_best.pct_change().dropna()
    sharpe_ratio_best = (daily_returns_best.mean() / daily_returns_best.std()) * np.sqrt(
        252) if daily_returns_best.std() != 0 else 0.0
    cumulative_max_best = portfolio_series_best.cummax()
    drawdowns_best = (portfolio_series_best - cumulative_max_best) / cumulative_max_best
    max_drawdown_best = drawdowns_best.min()

    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_series_best.index, portfolio_series_best.values, label='Portfolio Value', color='green')
    plt.title("Equity Curve (Best Hyperparameters)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.savefig(f"{plot_folder}/best_model_equity_curve.png")
    plt.close()

    plt.figure(figsize=(14, 7))
    plt.plot(drawdowns_best.index, drawdowns_best.values, label='Drawdown', color='red')
    plt.title("Drawdown Over Time (Best Hyperparameters)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.savefig(f"{plot_folder}/best_model_drawdown.png")
    plt.close()

    # Log final profit/loss and enhanced metrics
    log_and_print(f"Final Portfolio Value: ${final_value:.2f}")
    log_and_print(f"Total Profit/Loss: ${profit_loss:.2f}")
    log_and_print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    log_and_print(f"Max Drawdown: {max_drawdown:.4f}")

    # Save the best model again (redundant but ensures it's saved after all operations)
    best_model.save(f"{best_model_folder}/best_model.keras")
    log_and_print(f"Best model saved to '{best_model_folder}/best_model.keras'")
