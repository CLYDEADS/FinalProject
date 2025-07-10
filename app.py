from flask import Flask, redirect, url_for, request, flash, render_template, jsonify
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Integer, String, Text, DateTime, Float
from datetime import datetime
import pandas as pd
import joblib
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

# Initialize Flask app
app = Flask(__name__)
bootstrap = Bootstrap(app)

# Configuration
app.config['SECRET_KEY'] = 'secret123456'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Currency conversion - Static rate from NPR to PHP
NPR_TO_PHP_RATE = 0.40  # 1 Nepalese Rupee = 0.40 Philippine Peso

# Database setup
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
db.init_app(app)

# Models
class PredictionSearch(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, unique=True)
    commodity_name: Mapped[str] = mapped_column(String(100), nullable=False)
    forecast_days: Mapped[int] = mapped_column(Integer, nullable=False)
    search_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    avg_predicted_price: Mapped[float] = mapped_column(Float, nullable=True)

# Load model and data
MODEL_PATH = 'random_forest_model.pkl'
FEATURES_PATH = 'model_features.json'
DATA_PATH = 'kalimati_tarkari_dataset.csv'

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    model = None

# Load features list
try:
    with open(FEATURES_PATH, 'r') as f:
        features = json.load(f)
    print(f"Features loaded from {FEATURES_PATH}")
except FileNotFoundError:
    print(f"Error: Features file not found at {FEATURES_PATH}")
    features = []

# Load dataset
try:
    df_full_data = pd.read_csv(DATA_PATH)
    df_full_data['Date'] = pd.to_datetime(df_full_data['Date'], format='%Y-%m-%d')
    df_full_data = df_full_data.sort_values(by='Date')
    print(f"Dataset loaded from {DATA_PATH}")
    # Get unique commodities for the dropdown
    available_commodities = sorted(df_full_data['Commodity'].unique())
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATA_PATH}")
    df_full_data = None
    available_commodities = []

@app.route('/')
def index():
    """Main page showing search form and recent searches"""
    recent_searches = PredictionSearch.query.order_by(PredictionSearch.search_date.desc()).limit(10).all()
    return render_template('index.html', 
                         commodities=available_commodities, 
                         recent_searches=recent_searches)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if not model or df_full_data is None:
        flash('Prediction model or data not available', 'error')
        return redirect(url_for('index'))
    
    commodity_name = request.form['commodity_name']
    forecast_days = int(request.form['forecast_days'])
    
    # Validate inputs
    if commodity_name not in available_commodities:
        flash('Invalid commodity selected', 'error')
        return redirect(url_for('index'))
    
    if forecast_days < 1 or forecast_days > 90:
        flash('Forecast days must be between 1 and 90', 'error')
        return redirect(url_for('index'))
    
    try:
        # Generate prediction
        result = generate_prediction(commodity_name, forecast_days)
        
        if 'error' in result:
            flash(result['error'], 'error')
            return redirect(url_for('index'))
        
        # Calculate average predicted price (converted to PHP)
        avg_price = np.mean([float(item['price']) for item in result['forecast']])
        
        # Save search to database
        search_record = PredictionSearch(
            commodity_name=commodity_name,
            forecast_days=forecast_days,
            avg_predicted_price=avg_price  # Already converted in generate_prediction
        )
        db.session.add(search_record)
        db.session.commit()
        
        flash(f'Prediction generated for {commodity_name}', 'success')
        return render_template('results.html', 
                             commodity=commodity_name,
                             forecast_days=forecast_days,
                             forecast=result['forecast'],
                             plot_image=result['plot_image'])
        
    except Exception as e:
        flash(f'Error generating prediction: {str(e)}', 'error')
        return redirect(url_for('index'))

def generate_prediction(commodity_name, forecast_days):
    """Generate price prediction for given commodity and forecast days"""
    # Filter data for the selected commodity
    df_commodity = df_full_data[df_full_data['Commodity'] == commodity_name].copy()
    
    if df_commodity.empty:
        return {'error': f"No data found for commodity '{commodity_name}'."}
    
    # Sort by date
    df_commodity = df_commodity.sort_values(by='Date').reset_index(drop=True)
    
    # Feature engineering
    df_commodity['DayOfYear'] = df_commodity['Date'].dt.dayofyear
    df_commodity['WeekOfYear'] = df_commodity['Date'].dt.isocalendar().week.astype(int)
    df_commodity['Month'] = df_commodity['Date'].dt.month
    df_commodity['Year'] = df_commodity['Date'].dt.year
    df_commodity['DayOfWeek'] = df_commodity['Date'].dt.dayofweek
    
    # Lag features
    df_commodity['Average_Lag_1'] = df_commodity['Average'].shift(1)
    df_commodity['Average_Lag_7'] = df_commodity['Average'].shift(7)
    df_commodity['Average_Lag_30'] = df_commodity['Average'].shift(30)
    
    # Rolling statistics
    df_commodity['Average_MA_7'] = df_commodity['Average'].rolling(window=7).mean().shift(1)
    df_commodity['Average_STD_7'] = df_commodity['Average'].rolling(window=7).std().shift(1)
    
    # Drop NaN values
    df_commodity_processed = df_commodity.dropna().copy()
    
    if df_commodity_processed.empty:
        return {'error': f"Not enough historical data for '{commodity_name}' to generate features."}
    
    # Prepare features
    X = df_commodity_processed[features]
    y = df_commodity_processed['Average']
    
    # Split for visualization
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]
    
    # Get test predictions
    test_predictions = model.predict(X_test)
    
    # Future forecasting
    future_dates_forecast = []
    future_prices_forecast = []
    current_data = df_commodity_processed.copy()
    
    for i in range(forecast_days):
        if i == 0:
            current_date = current_data['Date'].max() + pd.Timedelta(days=1)
        else:
            current_date = future_dates_forecast[-1] + pd.Timedelta(days=1)
        
        # Extract date features
        day_of_year = current_date.dayofyear
        week_of_year = current_date.isocalendar().week
        month = current_date.month
        year = current_date.year
        day_of_week = current_date.dayofweek
        
        # Calculate lag features
        if future_prices_forecast:
            temp_series = pd.concat([
                current_data['Average'],
                pd.Series(future_prices_forecast)
            ]).reset_index(drop=True)
        else:
            temp_series = current_data['Average']
        
        avg_lag_1 = temp_series.iloc[-1] if len(temp_series) >= 1 else np.nan
        avg_lag_7 = temp_series.iloc[-7] if len(temp_series) >= 7 else np.nan
        avg_lag_30 = temp_series.iloc[-30] if len(temp_series) >= 30 else np.nan
        
        if len(temp_series) >= 7:
            avg_ma_7 = temp_series.rolling(window=7).mean().iloc[-1]
            avg_std_7 = temp_series.rolling(window=7).std().iloc[-1]
        else:
            avg_ma_7 = np.nan
            avg_std_7 = np.nan
        
        # Create feature row
        feature_data = {
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
        
        feature_df = pd.DataFrame([feature_data], columns=features)
        feature_df.dropna(inplace=True)
        
        if feature_df.empty:
            break
        
        predicted_price = model.predict(feature_df)[0]
        future_dates_forecast.append(current_date)
        future_prices_forecast.append(predicted_price)
    
    # Generate plot
    try:
        plt.ioff()  # Turn off interactive mode
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data (convert to PHP)
        ax.plot(df_commodity_processed['Date'][0:train_size], 
                df_commodity_processed['Average'][0:train_size] * NPR_TO_PHP_RATE, 
                label='Historical Prices (Train)', color='blue', alpha=0.7)
        
        # Plot test data (convert to PHP)
        ax.plot(df_commodity_processed['Date'][train_size:len(df_commodity_processed)], 
                y_test * NPR_TO_PHP_RATE, label='Test Actual', color='green', linewidth=2)
        
        ax.plot(df_commodity_processed['Date'][train_size:len(df_commodity_processed)], 
                test_predictions * NPR_TO_PHP_RATE, label='Test Predicted', color='red', linestyle='--')
        
        # Plot forecast (convert to PHP)
        ax.plot(future_dates_forecast, [p * NPR_TO_PHP_RATE for p in future_prices_forecast], 
                label=f'Forecast ({forecast_days} days)', color='purple', 
                linestyle='-', marker='o', markersize=3)
        
        ax.set_title(f'{commodity_name} Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Price (₱/Kg)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        
        # Convert plot to base64
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plot_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close(fig)  # Close the specific figure
        
    except Exception as e:
        print(f"Error generating plot: {e}")
        plot_base64 = None
    
    # Format results (convert NPR to PHP)
    forecast_results = [
        {'date': date.strftime('%Y-%m-%d'), 'price': f'{(price * NPR_TO_PHP_RATE):.2f}'}
        for date, price in zip(future_dates_forecast, future_prices_forecast)
    ]
    
    return {
        'forecast': forecast_results,
        'plot_image': plot_base64 if plot_base64 else ''
    }

@app.route('/history')
def history():
    """Show prediction history"""
    searches = PredictionSearch.query.order_by(PredictionSearch.search_date.desc()).all()
    return render_template('history.html', searches=searches)

@app.route('/delete_search/<int:search_id>')
def delete_search(search_id):
    """Delete a search record"""
    search = PredictionSearch.query.get_or_404(search_id)
    db.session.delete(search)
    db.session.commit()
    flash('Search record deleted', 'success')
    return redirect(url_for('history'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 