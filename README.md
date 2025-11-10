# High-Frequency Implied Volatility Forecasting

This project implements a baseline machine learning model to forecast 10-second-ahead implied volatility (IV) for Ethereum (ETH) using high-frequency, 1-second-resolution order book data. The model also incorporates cross-asset signals from other cryptocurrencies (like BTC, SOL, etc.) to improve predictive accuracy.

The model is built using `polars` for high-performance data manipulation and `LightGBM` for gradient boosting. The primary evaluation metric is the **Pearson Correlation Coefficient** between the predicted and actual IV.

## Problem Statement

The goal is to build and evaluate a time-series forecasting model that ingests 1-second-resolution order-book snapshots and cross-asset signals to produce implied volatility predictions at `t+10` seconds.

## Dataset

The data is expected to be in the following structure:

```
.
├── train/
│   ├── ETH.csv
│   ├── BTC.csv
│   ├── SOL.csv
│   └── ...
├── test/
│   ├── ETH.csv
│   ├── BTC.csv
│   ├── SOL.csv
│   └── ...
└── ...
```

### Data Columns

All order book CSVs (e.g., `ETH.csv`, `BTC.csv`) share these columns:

* `timestamp`: 1-second resolution timestamp (YYYY-MM-DD HH:MM:SS).
* `mid_price`: Mid-market price = (best bid + best ask) / 2.
* `bid_price1` … `bid_price5`: Price at bid levels 1-5.
* `bid_volume1` … `bid_volume5`: Volume at bid levels 1-5.
* `ask_price1` … `ask_price5`: Price at ask levels 1-5.
* `ask_volume1` … `ask_volume5`: Volume at ask levels 1-5.
* `label` (float): **Target variable**. The ground-truth 10-second-ahead implied volatility. (Only present in `train/ETH.csv`).

## Methodology

### 1. Feature Engineering

A variety of features are engineered from the raw order book data for the target asset (ETH) and all cross-assets (e.g., BTC, SOL):

* **Weighted Average Price (WAP)**: Captures the micro-price movement.
* **Order Book Imbalance (OBI)**: Measures the ratio of bid vs. ask volume at the top level.
* **Bid-Ask Spread**: The difference between the best ask and best bid.
* **Log Returns**: `log(mid_price)` differenced over 1 second.
* **Realized Volatility**: Rolling standard deviation of log returns over multiple windows (10s, 60s, 300s).
* **Total Imbalance**: Imbalance calculated across all 5 order book levels.
* **Lag Features**: Lagged values (1s, 5s, 10s, 30s) of WAP, OBI, Spread, and 10s Volatility.

### 2. Modeling

A **LightGBM (LGBM) Regressor** is used for its speed, efficiency, and high performance on large-scale tabular data.

### 3. Validation

The training data is split into training and validation sets using a time-based 80/20 split. The model is evaluated on the validation set using the **Pearson Correlation Coefficient** and **Mean Absolute Error (MAE)**.

This baseline achieves a validation score of **~0.75 Pearson Correlation**.

## How to Run



1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Pipeline:**
    The script will train the model, save it, and generate a `submission.csv` file.
    ```bash
    python train_predict.py
    ```

## Project Structure

```
.
├── train/                # Training data
│   ├── ETH.csv
│   └── ...
├── test/                 # Test data
│   ├── ETH.csv
│   └── ...
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── train_predict.py      # Main script for training and prediction
└── lgbm_volatility_model.joblib # (Generated) Saved model file
└── submission.csv        # (Generated) Final predictions
```