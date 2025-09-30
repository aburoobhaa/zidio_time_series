

Time Series Analysis & Forecasting for Stock Market (single-file)
- Timeline (Week numbers shifted: Week 2..12; Week 1 removed as requested)
- Fetches stock data (yfinance / optional CSV)
- Preprocesses, explores, fits ARIMA, Prophet, LSTM (Keras/TensorFlow)
- Compares models (RMSE, MAPE), saves plots & models

Usage:
    python project.py --ticker AAPL --start 2015-01-01 --end 2024-12-31 --steps all

Examples:
    python project.py --ticker AAPL --start 2018-01-01 --end 2024-01-01 --steps all

Dependencies:
    pip install yfinance pandas numpy matplotlib seaborn statsmodels prophet tensorflow scikit-learn
    (If prophet install issues, try: pip install prophet --no-cache-dir)
