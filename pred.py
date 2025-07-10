from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import json
import matplotlib.pyplot as plt
import io
import base64
from datetime import timedelta

# Initialize Flask app
app = Flask(__name__)

# --- Load the trained model and features list ---
# IMPORTANT: Update paths if your model/data are in different locations
MODEL_PATH = 'random_forest_model.pkl'
FEATURES_PATH = 'model_features.json'
DATA_PATH = 'kalimati_tarkari_dataset.csv' # Your dataset should be alongside app.py or accessible

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please ensure it's in the same directory or provide full path.")
    exit() # Exit if model not found

# Load features list
try:
    with open(FEATURES_PATH, 'r') as f:
        features = json.load(f)
    print(f"Features loaded from {FEATURES_PATH}")
except FileNotFoundError:
    print(f"Error: Features file not found at {FEATURES_PATH}. Please ensure it's in the same directory or provide full path.")
    exit()

# Load the full dataset (needed for feature engineering for future predictions)
try:
    df_full_data = pd.read_csv(DATA_PATH) # Renamed to df_full_data to avoid conflict with df_commodity
    df_full_data['Date'] = pd.to_datetime(df_full_data['Date'], format='%Y-%m-%d')
    df_full_data = df_full_data.sort_values(by='Date')
    print(f"Dataset loaded from {DATA_PATH}")
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATA_PATH}. Please ensure it's in the same directory or provide full path.")
    exit()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    # Get commodity from the request (from the HTML form)
    commodity_name = request.form['commodity_name']
    forecast_days = int(request.form['forecast_days']) # Get forecast days from form

    # Filter data for the selected commodity
    df_commodity = df_full_data[df_full_data['Commodity'] == commodity_name].copy()

    if df_commodity.empty:
        return jsonify({'error': f"No data found for commodity '{commodity_name}'."}), 400

    # Ensure df_commodity is sorted by date for correct feature engineering
    df_commodity = df_commodity.sort_values(by='Date').reset_index(drop=True)

    # Re-create features needed for historical data (like lags, rolling means) for the selected commodity
    # This must match your training feature engineering
    df_commodity['DayOfYear'] = df_commodity['Date'].dt.dayofyear
    df_commodity['WeekOfYear'] = df_commodity['Date'].dt.isocalendar().week.astype(int)
    df_commodity['Month'] = df_commodity['Date'].dt.month
    df_commodity['Year'] = df_commodity['Date'].dt.year
    df_commodity['DayOfWeek'] = df_commodity['Date'].dt.dayofweek

    df_commodity['Average_Lag_1'] = df_commodity['Average'].shift(1)
    df_commodity['Average_Lag_7'] = df_commodity['Average'].shift(7)
    df_commodity['Average_Lag_30'] = df_commodity['Average'].shift(30)

    df_commodity['Average_MA_7'] = df_commodity['Average'].rolling(window=7).mean().shift(1)
    df_commodity['Average_STD_7'] = df_commodity['Average'].rolling(window=7).std().shift(1)

    # Make a copy before dropping NaNs for test set split below
    df_commodity_processed = df_commodity.dropna().copy()

    if df_commodity_processed.empty:
        return jsonify({'error': f"Not enough historical data for '{commodity_name}' to generate features after dropping NaNs."}), 400

    # Define Features (X) and Target (y) using the processed data
    X = df_commodity_processed[features]
    y = df_commodity_processed['Average'] # 'Average' is the target

    # Split data into training and testing sets (time series split for visualization)
    train_size = int(len(X) * 0.8)
    # Ensure X_test and y_test are consistent with the dates for plotting
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    # Get predictions for the test set
    test_predictions = model.predict(X_test)


    # --- Iterative Forecasting Logic for future days ---
    future_dates_forecast = []
    future_prices_forecast = []

    # Use the processed df_commodity for starting point of future predictions
    current_data_for_future_prediction = df_commodity_processed.copy()


    for i in range(forecast_days):
        if i == 0:
            current_date_for_prediction = current_data_for_future_prediction['Date'].max() + pd.Timedelta(days=1)
        else:
            current_date_for_prediction = future_dates_forecast[-1] + pd.Timedelta(days=1)

        day_of_year = current_date_for_prediction.dayofyear
        week_of_year = current_date_for_prediction.isocalendar().week
        month = current_date_for_prediction.month
        year = current_date_for_prediction.year
        day_of_week = current_date_for_prediction.dayofweek

        if future_prices_forecast:
            temp_average_series = pd.concat([
                current_data_for_future_prediction['Average'],
                pd.Series(future_prices_forecast)
            ]).reset_index(drop=True)
        else:
            temp_average_series = current_data_for_future_prediction['Average']

        # Ensure enough data for lags/rolling window before calculating
        avg_lag_1 = temp_average_series.iloc[-1] if len(temp_average_series) >= 1 else np.nan
        avg_lag_7 = temp_average_series.iloc[-7] if len(temp_average_series) >= 7 else np.nan
        avg_lag_30 = temp_average_series.iloc[-30] if len(temp_average_series) >= 30 else np.nan

        if len(temp_average_series) >= 7:
            avg_ma_7 = temp_average_series.rolling(window=7).mean().iloc[-1]
            avg_std_7 = temp_average_series.rolling(window=7).std().iloc[-1]
        else:
            avg_ma_7 = np.nan
            avg_std_7 = np.nan

        current_future_features_data = {
            'DayOfYear': day_of_year,
            'WeekOfYear': week_of_year,
            'Month': month,
            'Year': year,
            'DayOfWeek': day_of_week,
            'Average_Lag_1': avg_lag_1,
            'Average_Lag_7': avg_lag_7,
            'Average_Lag_30': avg_lag_30,
            'Average_MA_7': avg_ma_7,
            'Average_STD_7': avg_std_7
        }

        current_future_X_df = pd.DataFrame([current_future_features_data], columns=features)
        current_future_X_df.dropna(inplace=True) # Drop if any NaN created here

        if current_future_X_df.empty:
            print(f"Skipping future prediction for {current_date_for_prediction.strftime('%Y-%m-%d')} due to missing features.")
            break

        predicted_price = model.predict(current_future_X_df)[0]

        future_dates_forecast.append(current_date_for_prediction)
        future_prices_forecast.append(predicted_price)

    # --- Generate Plot ---
    plt.figure(figsize=(16, 8))
    # Plot historical data (up to the split point)
    plt.plot(df_commodity_processed['Date'][0:train_size], df_commodity_processed['Average'][0:train_size], label='Historical Actual Prices (Train)', color='blue', alpha=0.7)
    # Plot test set actuals
    plt.plot(df_commodity_processed['Date'][train_size:len(df_commodity_processed)], y_test, label='Test Set Actual Prices', color='darkgreen', linestyle='-', linewidth=2)
    # Plot test set predictions
    plt.plot(df_commodity_processed['Date'][train_size:len(df_commodity_processed)], test_predictions, label='Test Set Predicted Prices', color='red', linestyle='--', alpha=0.7)
    # Plot future forecast
    plt.plot(future_dates_forecast, future_prices_forecast, label=f'Forecasted Prices ({forecast_days} days)', color='purple', linestyle='-', marker='o', markersize=4)

    plt.title(f'{commodity_name} Price Historicals, Test Predictions, and Future Forecast')
    plt.xlabel('Date')
    plt.ylabel('Average Price (Rs./Kg)')
    plt.legend()
    plt.grid(True)
    plt.axvline(x=df_commodity_processed['Date'].max(), color='gray', linestyle=':', label='End of Actual Data')
    plt.tight_layout() # Adjust plot to prevent labels overlapping

    # Save plot to a BytesIO object
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0) # Rewind the buffer to the beginning
    plot_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    plt.close() # Close the plot to free up memory

    # Format forecast results for display
    forecast_results = [
        {'date': date.strftime('%Y-%m-%d'), 'price': f'{price:.2f}'}
        for date, price in zip(future_dates_forecast, future_prices_forecast)
    ]

    # Return both forecast results and the plot image
    return jsonify({
        'forecast': forecast_results,
        'plot_image': plot_base64
    })

if __name__ == '__main__':
    # You'll run this locally first, then use a deployment method
    app.run(debug=True)