"""
Stock Market Time Series Analysis and Forecasting
Complete implementation with ARIMA, Prophet, and LSTM models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Data collection
import yfinance as yf

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet

# Deep Learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Visualization
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 6)


class StockForecaster:
    """Complete stock forecasting pipeline"""
    
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    # ==================== DATA COLLECTION ====================
    
    def fetch_data(self):
        """Download stock data from Yahoo Finance"""
        print(f"Fetching data for {self.ticker}...")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        print(f"Downloaded {len(self.data)} rows of data")
        print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
        return self.data
    
    # ==================== PREPROCESSING ====================
    
    def preprocess_data(self):
        """Clean and prepare data"""
        print("\nPreprocessing data...")
        
        # Handle missing values
        print(f"Missing values before: {self.data.isnull().sum().sum()}")
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)
        print(f"Missing values after: {self.data.isnull().sum().sum()}")
        
        # Add technical indicators
        self.data['MA7'] = self.data['Close'].rolling(window=7).mean()
        self.data['MA21'] = self.data['Close'].rolling(window=21).mean()
        self.data['MA50'] = self.data['Close'].rolling(window=50).mean()
        self.data['Volatility'] = self.data['Close'].rolling(window=21).std()
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        
        print("Technical indicators added: MA7, MA21, MA50, Volatility, Daily_Return")
        
    def resample_data(self, frequency='W'):
        """Resample data to different frequencies (D=daily, W=weekly, M=monthly)"""
        resampled = self.data.resample(frequency).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        return resampled
    
    # ==================== VISUALIZATION ====================
    
    def plot_overview(self, save_path='stock_overview.png'):
        """Comprehensive visualization of stock data"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Price and Moving Averages
        axes[0].plot(self.data.index, self.data['Close'], label='Close Price', linewidth=2)
        axes[0].plot(self.data.index, self.data['MA7'], label='7-Day MA', alpha=0.7)
        axes[0].plot(self.data.index, self.data['MA21'], label='21-Day MA', alpha=0.7)
        axes[0].plot(self.data.index, self.data['MA50'], label='50-Day MA', alpha=0.7)
        axes[0].set_title(f'{self.ticker} Stock Price Analysis', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volume
        axes[1].bar(self.data.index, self.data['Volume'], alpha=0.6, color='steelblue')
        axes[1].set_title('Trading Volume', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Volume')
        axes[1].grid(True, alpha=0.3)
        
        # Volatility
        axes[2].plot(self.data.index, self.data['Volatility'], color='red', alpha=0.7)
        axes[2].set_title('Price Volatility (21-Day Rolling Std)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Volatility')
        axes[2].grid(True, alpha=0.3)
        
        # Daily Returns
        axes[3].plot(self.data.index, self.data['Daily_Return'], color='green', alpha=0.6)
        axes[3].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[3].set_title('Daily Returns', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Return (%)')
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Overview plot saved to {save_path}")
        plt.show()
    
    def plot_seasonality(self, save_path='seasonality.png'):
        """Decompose time series into trend, seasonality, and residuals"""
        close_prices = self.data['Close'].dropna()
        decomposition = seasonal_decompose(close_prices, model='multiplicative', period=30)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 10))
        
        decomposition.observed.plot(ax=axes[0], title='Observed', color='blue')
        decomposition.trend.plot(ax=axes[1], title='Trend', color='green')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonality', color='red')
        decomposition.resid.plot(ax=axes[3], title='Residuals', color='purple')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Seasonality plot saved to {save_path}")
        plt.show()
    
    # ==================== ARIMA/SARIMA MODELS ====================
    
    def train_arima(self, order=(5,1,2), forecast_days=30):
        """Train ARIMA model"""
        print(f"\nTraining ARIMA model with order={order}...")
        
        train_data = self.data['Close'][:-forecast_days]
        test_data = self.data['Close'][-forecast_days:]
        
        model = ARIMA(train_data, order=order)
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=forecast_days)
        
        self.models['ARIMA'] = fitted_model
        self.predictions['ARIMA'] = forecast
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(test_data, forecast))
        mape = mean_absolute_percentage_error(test_data, forecast) * 100
        
        self.metrics['ARIMA'] = {'RMSE': rmse, 'MAPE': mape}
        
        print(f"ARIMA - RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        return fitted_model, forecast
    
    def train_sarima(self, order=(1,1,1), seasonal_order=(1,1,1,12), forecast_days=30):
        """Train SARIMA model"""
        print(f"\nTraining SARIMA model with order={order}, seasonal_order={seasonal_order}...")
        
        train_data = self.data['Close'][:-forecast_days]
        test_data = self.data['Close'][-forecast_days:]
        
        model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        
        # Forecast
        forecast = fitted_model.forecast(steps=forecast_days)
        
        self.models['SARIMA'] = fitted_model
        self.predictions['SARIMA'] = forecast
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(test_data, forecast))
        mape = mean_absolute_percentage_error(test_data, forecast) * 100
        
        self.metrics['SARIMA'] = {'RMSE': rmse, 'MAPE': mape}
        
        print(f"SARIMA - RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        return fitted_model, forecast
    
    # ==================== PROPHET MODEL ====================
    
    def train_prophet(self, forecast_days=30):
        """Train Facebook Prophet model"""
        print("\nTraining Prophet model...")
        
        # Prepare data for Prophet
        df_prophet = pd.DataFrame({
            'ds': self.data.index,
            'y': self.data['Close'].values
        })
        
        train_data = df_prophet[:-forecast_days]
        test_data = df_prophet[-forecast_days:]
        
        # Train model
        model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
        model.fit(train_data)
        
        # Forecast
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        
        predicted_values = forecast['yhat'][-forecast_days:].values
        
        self.models['Prophet'] = model
        self.predictions['Prophet'] = predicted_values
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(test_data['y'], predicted_values))
        mape = mean_absolute_percentage_error(test_data['y'], predicted_values) * 100
        
        self.metrics['Prophet'] = {'RMSE': rmse, 'MAPE': mape}
        
        print(f"Prophet - RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        return model, predicted_values
    
    # ==================== LSTM MODEL ====================
    
    def prepare_lstm_data(self, lookback=60, forecast_days=30):
        """Prepare data for LSTM model"""
        close_prices = self.data['Close'].values.reshape(-1, 1)
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data) - forecast_days):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Split into train and test
        split_idx = len(X) - forecast_days
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test, scaler
    
    def train_lstm(self, lookback=60, forecast_days=30, epochs=50, batch_size=32):
        """Train LSTM neural network"""
        print(f"\nTraining LSTM model with lookback={lookback}, epochs={epochs}...")
        
        X_train, X_test, y_train, y_test, scaler = self.prepare_lstm_data(lookback, forecast_days)
        
        # Build LSTM model
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Predict
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        self.models['LSTM'] = {'model': model, 'scaler': scaler, 'lookback': lookback}
        self.predictions['LSTM'] = predictions.flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
        mape = mean_absolute_percentage_error(y_test_actual, predictions) * 100
        
        self.metrics['LSTM'] = {'RMSE': rmse, 'MAPE': mape}
        
        print(f"LSTM - RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        return model, predictions, history
    
    # ==================== EVALUATION & COMPARISON ====================
    
    def plot_predictions(self, forecast_days=30, save_path='predictions_comparison.png'):
        """Plot actual vs predicted prices for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        actual_prices = self.data['Close'][-forecast_days:].values
        test_dates = self.data.index[-forecast_days:]
        
        models_to_plot = [('ARIMA', 0, 0), ('SARIMA', 0, 1), 
                          ('Prophet', 1, 0), ('LSTM', 1, 1)]
        
        for model_name, row, col in models_to_plot:
            if model_name in self.predictions:
                ax = axes[row, col]
                
                ax.plot(test_dates, actual_prices, label='Actual', 
                       color='black', linewidth=2, marker='o', markersize=4)
                ax.plot(test_dates, self.predictions[model_name], 
                       label='Predicted', color='red', linewidth=2, 
                       marker='s', markersize=4, alpha=0.7)
                
                metrics = self.metrics[model_name]
                ax.set_title(f'{model_name} Predictions\nRMSE: {metrics["RMSE"]:.2f}, MAPE: {metrics["MAPE"]:.2f}%',
                           fontsize=12, fontweight='bold')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPredictions comparison plot saved to {save_path}")
        plt.show()
    
    def generate_report(self, save_path='evaluation_report.txt'):
        """Generate comprehensive evaluation report"""
        report = []
        report.append("=" * 70)
        report.append(f"STOCK MARKET FORECASTING EVALUATION REPORT")
        report.append(f"Ticker: {self.ticker}")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        report.append("")
        
        report.append("MODEL PERFORMANCE COMPARISON")
        report.append("-" * 70)
        report.append(f"{'Model':<15} {'RMSE':<15} {'MAPE (%)':<15} {'Rank':<10}")
        report.append("-" * 70)
        
        # Sort by RMSE
        sorted_models = sorted(self.metrics.items(), key=lambda x: x[1]['RMSE'])
        
        for rank, (model_name, metrics) in enumerate(sorted_models, 1):
            report.append(f"{model_name:<15} {metrics['RMSE']:<15.2f} {metrics['MAPE']:<15.2f} {rank:<10}")
        
        report.append("-" * 70)
        report.append(f"\nBest Model: {sorted_models[0][0]} (Lowest RMSE)")
        report.append("")
        
        report.append("RECOMMENDATIONS")
        report.append("-" * 70)
        best_model = sorted_models[0][0]
        
        if best_model == 'LSTM':
            report.append("• LSTM performed best - Deep learning captures complex patterns")
            report.append("• Consider: Longer training, more features, hyperparameter tuning")
        elif best_model == 'Prophet':
            report.append("• Prophet performed best - Good for trend and seasonality")
            report.append("• Consider: Adding custom seasonalities and holidays")
        elif best_model in ['ARIMA', 'SARIMA']:
            report.append(f"• {best_model} performed best - Classical methods work well")
            report.append("• Consider: Grid search for optimal parameters")
        
        report.append("")
        report.append("DATA SUMMARY")
        report.append("-" * 70)
        report.append(f"Total data points: {len(self.data)}")
        report.append(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        report.append(f"Mean Close Price: ${self.data['Close'].mean():.2f}")
        report.append(f"Price Range: ${self.data['Close'].min():.2f} - ${self.data['Close'].max():.2f}")
        report.append(f"Average Daily Return: {self.data['Daily_Return'].mean()*100:.3f}%")
        report.append(f"Volatility (Std): {self.data['Close'].std():.2f}")
        report.append("")
        
        report_text = "\n".join(report)
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport saved to {save_path}")
        
        return report_text
    
    # ==================== MODEL PERSISTENCE ====================
    
    def save_models(self, directory='saved_models'):
        """Save trained models"""
        import os
        import pickle
        
        os.makedirs(directory, exist_ok=True)
        
        # Save ARIMA/SARIMA
        if 'ARIMA' in self.models:
            self.models['ARIMA'].save(f'{directory}/arima_model.pkl')
            print(f"ARIMA model saved to {directory}/arima_model.pkl")
        
        if 'SARIMA' in self.models:
            self.models['SARIMA'].save(f'{directory}/sarima_model.pkl')
            print(f"SARIMA model saved to {directory}/sarima_model.pkl")
        
        # Save Prophet
        if 'Prophet' in self.models:
            with open(f'{directory}/prophet_model.pkl', 'wb') as f:
                pickle.dump(self.models['Prophet'], f)
            print(f"Prophet model saved to {directory}/prophet_model.pkl")
        
        # Save LSTM
        if 'LSTM' in self.models:
            self.models['LSTM']['model'].save(f'{directory}/lstm_model.h5')
            with open(f'{directory}/lstm_scaler.pkl', 'wb') as f:
                pickle.dump(self.models['LSTM']['scaler'], f)
            print(f"LSTM model saved to {directory}/lstm_model.h5")
        
        print(f"\nAll models saved to {directory}/")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Configuration
    TICKER = 'AAPL'  # Change to any stock ticker (AAPL, GOOGL, MSFT, TSLA, etc.)
    START_DATE = '2020-01-01'
    END_DATE = '2024-12-31'
    FORECAST_DAYS = 30
    
    print("=" * 70)
    print("STOCK MARKET TIME SERIES FORECASTING")
    print("=" * 70)
    
    # Initialize forecaster
    forecaster = StockForecaster(TICKER, START_DATE, END_DATE)
    
    # Step 1: Data Collection
    forecaster.fetch_data()
    
    # Step 2: Preprocessing & Visualization
    forecaster.preprocess_data()
    forecaster.plot_overview()
    forecaster.plot_seasonality()
    
    # Step 3: Train Models
    forecaster.train_arima(order=(5,1,2), forecast_days=FORECAST_DAYS)
    forecaster.train_sarima(order=(1,1,1), seasonal_order=(1,1,1,7), forecast_days=FORECAST_DAYS)
    forecaster.train_prophet(forecast_days=FORECAST_DAYS)
    forecaster.train_lstm(lookback=60, forecast_days=FORECAST_DAYS, epochs=50)
    
    # Step 4: Evaluation
    forecaster.plot_predictions(forecast_days=FORECAST_DAYS)
    forecaster.generate_report()
    
    # Step 5: Save Models
    forecaster.save_models()
    
    print("\n" + "=" * 70)
    print("FORECASTING COMPLETE!")
    print("=" * 70)
