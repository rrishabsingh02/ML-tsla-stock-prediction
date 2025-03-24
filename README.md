# ML-tsla-stock-prediction - Group 24

# 🚀 TSLA LSTM Trading Agent

## 📈 Project Goal

This project builds a machine learning-based trading agent that predicts **daily Buy/Sell/Hold signals** for **Tesla (TSLA)** stock. The model uses a Long Short-Term Memory (LSTM) neural network trained on engineered technical indicators. It is designed to make **real-time decisions at 10:00 AM EST each trading day**, based on the previous 60 days of market data.

---

## 🧠 LSTM-Based Classification Approach

The model uses an LSTM architecture to capture **temporal patterns in technical features** and make 3-way predictions:

- **Buy (1)**
- **Hold (0)**
- **Sell (-1)**

The model is trained using **supervised learning**, with outcome-based labels reflecting expected price movement.

---

## 📁 Datasets

| File            | Source        | Date Range              | Purpose           |
|-----------------|---------------|--------------------------|--------------------|
| `TSLA_1yr.csv`  | Yahoo Finance | Mar 21, 2022 – Mar 14, 2025 | Model training     |
| `TSLA_test.csv` | Yahoo Finance | Mar 17, 2025 – Mar 21, 2025 | Final evaluation   |

---

## ⚙️ Feature Engineering

The model uses a curated set of **technical indicators** and **binary logic flags** based on common trading patterns.

### 📌 Technical Indicators

- `RSI`: 14-period Relative Strength Index
- `MACD_Line`: EMA(7) – EMA(14)
- `Signal_Line`: 9-period EMA of MACD
- `MACD_Histogram`: MACD_Line – Signal_Line

### 📌 Logic Flags

| Feature               | Description |
|------------------------|-------------|
| `MACD_Cross_Up`        | MACD crosses above Signal Line |
| `MACD_Cross_Down`      | MACD crosses below Signal Line |
| `MACD_Trending_Up`     | MACD and Signal both increasing |
| `MACD_Trending_Down`   | MACD and Signal both decreasing |
| `RSI_Entry_Zone`       | RSI < 45 (Buy zone) |
| `RSI_Exit_Zone`        | RSI > 65 (Sell zone) |

All features are normalized using `MinMaxScaler`.

---

## 🏷️ Labeling Strategy

Labels are based on **future price movement** using a 1-day lookahead window:

- **Buy (1)** → if return > **+2.0%**
- **Sell (-1)** → if return < **-2.0%**
- **Hold (0)** → otherwise

This labeling method allows the model to learn trade-worthy setups based on historical profitability.

---

## 🧪 Model Training Details

| Setting             | Value              |
|---------------------|--------------------|
| Model Type          | LSTM (2 layers)    |
| Sequence Length     | 60 days            |
| Loss Function       | Categorical Crossentropy |
| Optimizer           | Adam (lr = 0.0005) |
| Dropout             | 0.2 – 0.3          |
| Early Stopping      | Yes (patience = 5) |
| Class Weighting     | Yes (balanced)     |

---

## 📤 Real-Time Prediction Flow

Each trading morning at **10:00 AM EST**:

1. Fetch the latest 60 days of TSLA data
2. Compute features and logic flags
3. Scale using the trained MinMaxScaler
4. Predict with the LSTM model
5. Take action based on output:

```python
if prediction == 1 and confidence > 0.7:
    # Buy
elif prediction == -1 and confidence > 0.7:
    # Sell
else:
    # Hold
