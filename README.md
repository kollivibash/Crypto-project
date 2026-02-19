# 📈 Ethereum Price Prediction using XGBoost

A machine learning project that predicts **Ethereum (ETH-USD) closing prices** using historical market data and an XGBoost regression model.

This project demonstrates **time-series forecasting**, feature scaling, sliding window modeling, and recursive multi-step prediction.

---

## 🚀 Project Overview

This repository implements:

* 📥 Historical data download using `yfinance`
* 🧹 Data preprocessing & normalization
* 🔁 Sliding window time-series transformation
* 🌳 XGBoost regression model
* 📊 Actual vs Predicted price visualization
* 🔮 10-day future price forecasting

The model is trained on Ethereum daily closing prices from **2017 to 2025**.

---

## 🛠 Tech Stack

* Python 3.x
* pandas
* numpy
* yfinance
* scikit-learn
* XGBoost
* Plotly

---

## 📂 Project Structure

```
ethereum-price-prediction/
│
├── ethereum_prediction.py
├── requirements.txt
└── README.md
```

---

## 📥 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ethereum-price-prediction.git
cd ethereum-price-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy yfinance scikit-learn xgboost plotly
```

---

## ▶️ How It Works

### 1️⃣ Download Data

Uses Yahoo Finance API to download ETH-USD historical data:

```python
df = yf.download("ETH-USD", start="2017-01-01", end="2025-01-01")
```

---

### 2️⃣ Preprocessing

* Selects closing price
* Scales data using `MinMaxScaler`
* Splits into training (70%) and testing (30%)

---

### 3️⃣ Time-Series Windowing

Creates supervised learning dataset using 15-day lookback window:

```
[Day1 ... Day15] → Day16
```

---

### 4️⃣ Model Training

Uses XGBoost Regressor:

```python
XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6
)
```

---

### 5️⃣ Evaluation Metrics

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)

---

### 6️⃣ Future Forecasting

Predicts the next 10 days recursively using model outputs as inputs.

---

## 📊 Sample Output

* Interactive Plotly graph showing:

  * Actual price
  * Predicted price

* Console output:

  ```
  MAE: 0.0234
  RMSE: 0.0456

  Next 10 Days Prediction:
  [3200.45, 3225.10, 3250.87, ...]
  ```

*(Values will vary based on market conditions)*

---

## ⚠️ Limitations

* Uses only closing price (no technical indicators)
* Recursive forecasting can accumulate prediction errors
* Crypto markets are highly volatile
* Not intended for financial advice

---

## 📈 Possible Improvements

* Add technical indicators (SMA, EMA, RSI, Volatility)
* Implement walk-forward validation
* Use LSTM / GRU for deep learning
* Hyperparameter tuning with Optuna
* Add trading strategy backtesting
* Deploy as REST API (FastAPI)

---

## 📌 Future Roadmap

* [ ] Add multi-feature training
* [ ] Add directional accuracy metric
* [ ] Add feature importance visualization
* [ ] Add live price prediction mode
* [ ] Dockerize project

---

## 🧠 Learning Outcomes

This project demonstrates:

* Time-series data transformation
* Tree-based regression for financial forecasting
* Multi-step prediction logic
* Model evaluation for regression problems

---

## 📜 Disclaimer

This project is for **educational and research purposes only**.
Cryptocurrency trading involves significant financial risk.
Do not use predictions for real investment decisions without proper financial consultation.

---

## 👤 Author

**KOLLI VIBASH**
Machine Learning | Time Series | Quantitative Finance

GitHub: https://github.com/kollivibash

---

If you found this useful, ⭐ star the repository!
