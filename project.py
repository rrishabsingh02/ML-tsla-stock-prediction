import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Constants
START_CAPITAL = 10000
TRANSACTION_FEE = 0.01
DATA_PATH = "C:/Users/saiee/Downloads/Teslastockprediction/TSLA.csv"

# Load historical Tesla stock data
def get_stock_data():
    data = pd.read_csv(DATA_PATH)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data['Returns'] = data['Adj Close'].pct_change()
    return data.dropna()

# Feature engineering
def create_features(data):
    data['SMA_10'] = data['Adj Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Adj Close'].rolling(window=50).mean()
    data['Volatility'] = data['Returns'].rolling(window=10).std()
    data['Momentum'] = data['Adj Close'].diff(5)
    data['Target'] = np.where(data['Returns'].shift(-1) > 0, 1, 0)
    return data.dropna()

# Train ML model
def train_model(data):
    features = ['SMA_10', 'SMA_50', 'Volatility', 'Momentum']
    X = data[features]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    accuracy = model.score(X_test_scaled, y_test)
    print(f'Model Accuracy: {accuracy:.2f}')
    
    return model, scaler

# Trading strategy
def make_trade_decision(model, scaler, latest_data):
    features = ['SMA_10', 'SMA_50', 'Volatility', 'Momentum']
    scaled_data = scaler.transform(latest_data[features].values.reshape(1, -1))
    prediction = model.predict(scaled_data)[0]
    return 'Buy' if prediction == 1 else 'Sell'

# Execute simulation
def run_simulation():
    data = get_stock_data()
    data = create_features(data)
    model, scaler = train_model(data)
    
    capital = START_CAPITAL
    shares = 0
    
    trading_days = data.index[-5:].tolist()  # Last 5 available trading days in dataset
    
    for day in trading_days:
        latest_data = data.loc[data.index < day].iloc[-1]
        decision = make_trade_decision(model, scaler, latest_data)
        price = data.loc[day, 'Adj Close']
        
        if decision == 'Buy' and capital > 0:
            shares = (capital * (1 - TRANSACTION_FEE)) / price
            capital = 0
        elif decision == 'Sell' and shares > 0:
            capital = shares * price * (1 - TRANSACTION_FEE)
            shares = 0
        
        print(f'{day.date()}: {decision} at ${price:.2f} | Capital: ${capital:.2f} | Shares: {shares:.2f}')
    
    final_balance = capital + (shares * price)
    print(f'Final Account Balance: ${final_balance:.2f}')

# Run simulation
run_simulation()
