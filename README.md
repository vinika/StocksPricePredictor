📈 Stock Price Prediction with Machine Learning
  
   
A comprehensive machine learning project that predicts stock prices using three different approaches: LSTM (Deep Learning), XGBoost (Ensemble Learning), and ARIMA (Statistical Time Series).
Author: Vinika Gupta | LinkedIn | vinika03@gmail.com
 
🎯 Project Overview
This project demonstrates end-to-end machine learning pipeline for financial time series forecasting, including:
•	Data Collection: Real-time stock data from Yahoo Finance API
•	Feature Engineering: 12+ technical indicators (SMA, RSI, MACD, Bollinger Bands, etc.)
•	Model Training: Three different ML approaches with performance comparison
•	Prediction: 30-day future price forecasts
•	Visualization: Interactive charts and comprehensive analysis dashboard
 
🚀 Features
Machine Learning Models
•	LSTM (Long Short-Term Memory): Deep learning model capturing sequential patterns
•	XGBoost: Gradient boosting with engineered features for robust predictions
•	ARIMA: Statistical time series model for trend analysis
Technical Analysis
•	Moving Averages (SMA 20, SMA 50, EMA 20)
•	Relative Strength Index (RSI)
•	Moving Average Convergence Divergence (MACD)
•	Bollinger Bands
•	Momentum Indicators
•	Volatility Analysis
Metrics & Evaluation
•	Root Mean Squared Error (RMSE)
•	Mean Absolute Error (MAE)
•	R² Score
•	Model Performance Comparison
•	Annualized Volatility
•	Trend Detection (Bullish/Bearish)
 
📋 Prerequisites
System Requirements
•	Python 3.8 or higher
•	Jupyter Notebook or JupyterLab
•	4GB+ RAM recommended for LSTM training
Required Libraries
yfinance          # Stock data retrieval
numpy             # Numerical computations
pandas            # Data manipulation
matplotlib        # Static visualizations
seaborn           # Statistical visualizations
scikit-learn      # ML utilities and metrics
xgboost           # Gradient boosting
tensorflow        # Deep learning (LSTM)
statsmodels       # Time series (ARIMA)
plotly            # Interactive visualizations
 
🛠️ Installation
Option 1: Quick Install
pip install yfinance numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow statsmodels plotly
Option 2: Using Virtual Environment (Recommended)
Windows:
# Create virtual environment
python -m venv stock_env

# Activate environment
stock_env\Scripts\activate

# Install dependencies
pip install yfinance numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow statsmodels plotly
Mac/Linux:
# Create virtual environment
python3 -m venv stock_env

# Activate environment
source stock_env/bin/activate

# Install dependencies
pip install yfinance numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow statsmodels plotly
Option 3: Using Conda
# Create conda environment
conda create -n stock_pred python=3.10 -y

# Activate environment
conda activate stock_pred

# Install dependencies
pip install yfinance numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow statsmodels plotly
Verify Installation
Run this in a Jupyter cell:
import yfinance, numpy, pandas, matplotlib, seaborn, sklearn, xgboost, tensorflow, statsmodels, plotly
print("✅ All libraries installed successfully!")
 
📊 Usage
Basic Usage
1.	Clone or download the notebook
2.	Open Jupyter Notebook: 
3.	jupyter notebook
4.	Run all cells sequentially (Shift + Enter)
Customization
Change Stock Ticker
TICKER = 'AAPL'  # Change to: MSFT, GOOGL, TSLA, AMZN, etc.
Adjust Date Range
START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
Modify Prediction Horizon
PREDICTION_DAYS = 30  # Change to: 7, 60, 90 days
Train-Test Split
TRAIN_TEST_SPLIT = 0.8  # 80% training, 20% testing
Model Hyperparameters
LSTM Configuration
# In build_lstm_model() function
units=50          # Number of LSTM units
dropout=0.2       # Dropout rate
epochs=50         # Training epochs
batch_size=32     # Batch size
XGBoost Configuration
# In train_xgboost() function
n_estimators=100  # Number of trees
max_depth=5       # Tree depth
learning_rate=0.1 # Learning rate
ARIMA Configuration
# In train_arima() function
order=(5, 1, 0)   # (p, d, q) parameters
 
📁 Project Structure
stock-price-prediction/
│
├── stock_prediction.ipynb     # Main Jupyter notebook
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
└── outputs/                    # Generated outputs
    ├── AAPL_predictions_YYYYMMDD.csv
    ├── model_comparison.png
    └── feature_importance.png
 
🎓 How It Works
1. Data Collection
•	Fetches historical stock data from Yahoo Finance
•	Date range: Configurable (default: 2020-present)
•	Includes: Open, High, Low, Close, Volume
2. Feature Engineering
Generates 12 technical indicators:
•	Trend: SMA (20, 50), EMA (20)
•	Momentum: RSI, MACD, Signal Line
•	Volatility: Bollinger Bands, Standard Deviation
•	Other: Volume, Daily Returns, Momentum
3. Model Training
LSTM (Deep Learning)
•	3-layer LSTM architecture with dropout
•	60-day lookback window
•	Trained on normalized data (MinMaxScaler)
XGBoost (Gradient Boosting)
•	Uses 8 engineered features
•	100 decision trees
•	Feature importance analysis included
ARIMA (Statistical)
•	AutoRegressive Integrated Moving Average
•	Order (5,1,0) - configurable
•	Statistical time series approach
4. Evaluation & Comparison
•	Metrics: RMSE, MAE, R²
•	Visual comparison charts
•	Performance benchmarking
5. Prediction & Visualization
•	30-day future forecasts
•	Interactive Plotly charts
•	Exportable CSV results
 
📈 Sample Output
Model Performance Comparison
============================================================
📊 MODEL PERFORMANCE COMPARISON
============================================================
    Model    RMSE    MAE  R² Score
     LSTM    2.34   1.87     0.924
  XGBoost    2.89   2.15     0.891
    ARIMA    3.12   2.43     0.856
============================================================
Market Metrics
📊 Market Metrics:
   Volatility: 28.45%
   Trend: Strong Bullish (6.23%)
Output Files
•	AAPL_predictions_20241016.csv - Forecast data
•	Interactive Plotly charts (in notebook)
•	Model comparison visualizations
 
🎨 Visualizations
The notebook generates:
1.	Historical Price Chart - Candlestick with volume
2.	LSTM Training History - Loss curves
3.	XGBoost Feature Importance - Top contributing features
4.	30-Day Forecast Chart - All three models compared
5.	Model Comparison Bar Charts - RMSE, MAE, R² metrics
 
🔧 Troubleshooting
Common Issues
Issue 1: Yahoo Finance Download Error
Error: No data found, symbol may be delisted
Solution:
# Verify ticker symbol is correct
# Check if market is open
# Try a different ticker (e.g., 'MSFT', 'GOOGL')
Issue 2: TensorFlow Import Error
Error: ImportError: cannot import name 'Sequential'
Solution:
import tensorflow as tf
from tensorflow import keras
Sequential = keras.models.Sequential
Issue 3: Plotly Not Displaying
Solution:
import plotly.io as pio
pio.renderers.default = "notebook"  # or "jupyterlab"
Issue 4: Memory Error During LSTM Training
Solution:
# Reduce batch size
batch_size=16  # instead of 32

# Or reduce epochs
epochs=25  # instead of 50
Issue 5: Multi-level Column Headers
Already fixed in code - automatically flattens yfinance multi-level columns
 
🚀 Advanced Usage
Backtesting
Add this code to test historical predictions:
# Split data into multiple time periods
# Train on earlier data
# Test on later data
# Calculate accuracy over time
Ensemble Predictions
Combine all three models:
ensemble_prediction = (lstm_pred * 0.4 + 
                       xgb_pred * 0.35 + 
                       arima_pred * 0.25)
Add More Features
# Sentiment analysis from news
# Market indices (S&P 500, NASDAQ)
# Sector performance
# Economic indicators
Real-time Prediction API
# Deploy model using Flask/FastAPI
# Create REST endpoint for predictions
# Integrate with trading platforms
 
📚 Technical Details
LSTM Architecture
Layer 1: LSTM(50 units, return_sequences=True)
Dropout: 0.2
Layer 2: LSTM(50 units, return_sequences=True)
Dropout: 0.2
Layer 3: LSTM(50 units)
Dropout: 0.2
Output: Dense(1)
Optimizer: Adam (lr=0.001)
Loss: MSE
Feature Engineering Formulas
RSI (Relative Strength Index)
RS = Average Gain / Average Loss
RSI = 100 - (100 / (1 + RS))
MACD
MACD = EMA(12) - EMA(26)
Signal = EMA(9) of MACD
Bollinger Bands
Middle = SMA(20)
Upper = Middle + (2 × STD(20))
Lower = Middle - (2 × STD(20))
 
🎯 Use Cases
•	Quantitative Finance Research: Evaluate different forecasting approaches
•	Portfolio Management: Risk assessment and trend analysis
•	Algorithmic Trading: Feature engineering for trading strategies
•	Academic Projects: Time series analysis and ML applications
•	Learning: Hands-on experience with financial ML
 
📝 Future Enhancements
•	[ ] Real-time prediction dashboard
•	[ ] Sentiment analysis integration (Twitter, News)
•	[ ] Multi-stock portfolio optimization
•	[ ] Risk metrics (Sharpe Ratio, VaR, Max Drawdown)
•	[ ] Automated trading backtesting
•	[ ] Model deployment with Flask API
•	[ ] Transformer-based models (Attention mechanisms)
•	[ ] Options pricing prediction
 
🤝 Contributing
Contributions are welcome! Please feel free to:
1.	Fork the repository
2.	Create a feature branch
3.	Submit a pull request
Areas for Contribution
•	Additional ML models (Prophet, GRU, Transformer)
•	More technical indicators
•	Enhanced visualizations
•	Performance optimizations
•	Documentation improvements
 
⚠️ Disclaimer
IMPORTANT: This project is for educational and research purposes only.
•	Not financial advice
•	Past performance doesn't guarantee future results
•	Stock markets are inherently unpredictable
•	Always consult with financial advisors before trading
•	Use at your own risk
 
📜 License
This project is licensed under the MIT License.
MIT License

Copyright (c) 2025 Vinika Gupta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 
🌟 Acknowledgments
•	Yahoo Finance for providing free stock data API
•	TensorFlow/Keras for deep learning framework
•	XGBoost developers for gradient boosting library
•	statsmodels for time series analysis tools
•	Plotly for interactive visualizations
•	CLAUDE.AI for code related help
 
📞 Contact
Vinika Gupta
•	💼 LinkedIn: linkedin.com/in/vinika-gupta
 
🎓 About the Author
Data Scientist with 5+ years of experience in machine learning, deep learning, and statistical modeling. Specializes in time series analysis, NLP, and product-focused analytics. Previous experience at Nordstrom (Search Relevance), Auburn University (Computer Vision Research), and Siemens (Energy Analytics).
Skills: Python, TensorFlow, XGBoost, AWS, Time Series Analysis, A/B Testing, Statistical Modeling
 
⭐ Show Your Support
If you found this project helpful, please consider:
•	Giving it a ⭐ star
•	Sharing with others
•	Contributing improvements
•	Providing feedback
 
Happy Predicting! 📈🚀
Last Updated: October 2025



