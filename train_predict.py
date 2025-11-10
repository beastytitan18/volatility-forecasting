import logging
import gc
from pathlib import Path
from typing import List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import pearsonr
import joblib

# --- 1. Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# --- 2. Configuration ---
class Config:
    """
    Configuration class for all script parameters.
    """
    # --- Paths ---
    TRAIN_DIR = Path("train/")
    TEST_DIR = Path("test/")
    MODEL_OUTPUT_PATH = Path("lgbm_volatility_model.joblib")
    SUBMISSION_PATH = Path("submission.csv")

    # --- Data ---
    TARGET_ASSET = "ETH"
    # Use a specific list or set to None to use all available assets
    CROSS_ASSETS = ["BTC", "SOL"] 
    # CROSS_ASSETS = None # Uncomment to use all assets in the dir

    # Columns to be read and cast to Float32
    ALL_COLS = [
        "mid_price", "bid_price1", "bid_volume1", "bid_price2", "bid_volume2",
        "bid_price3", "bid_volume3", "bid_price4", "bid_volume4", "bid_price5",
        "bid_volume5", "ask_price1", "ask_volume1", "ask_price2", "ask_volume2",
        "ask_price3", "ask_volume3", "ask_price4", "ask_volume4", "ask_price5",
        "ask_volume5"
    ]
    
    @staticmethod
    def get_schema_overrides() -> dict:
        """Generates schema overrides for Polars CSV reader."""
        schema = {col: pl.Float32 for col in Config.ALL_COLS}
        schema["label"] = pl.Float32
        return schema

    # --- Model ---
    TARGET_COL = "label"
    VALIDATION_SPLIT_RATIO = 0.8
    
    LGBM_PARAMS = {
        'objective': 'regression_l1',  # MAE
        'metric': 'l1',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
    }
    
    LGBM_FIT_PARAMS = {
        "eval_metric": "l1",
        "callbacks": [lgb.early_stopping(100, verbose=True)],
    }


# --- 3. Feature Engineering ---
def create_orderbook_features(df: pl.DataFrame, asset_name: str = "") -> pl.DataFrame:
    """
    Engineers a set of features from the raw order book data for a single asset.
    """
    prefix = f"{asset_name}_" if asset_name else ""

    # Calculate Weighted Average Price (WAP)
    wap1 = (
        (pl.col("bid_price1") * pl.col("ask_volume1") + pl.col("ask_price1") * pl.col("bid_volume1")) /
        (pl.col("bid_volume1") + pl.col("ask_volume1"))
    )
    
    # Calculate Order Book Imbalance (OBI)
    obi1 = (
        (pl.col("bid_volume1") - pl.col("ask_volume1")) /
        (pl.col("bid_volume1") + pl.col("ask_volume1"))
    )
    
    # Calculate Bid-Ask Spread
    spread1 = (pl.col("ask_price1") - pl.col("bid_price1"))

    # Calculate log returns of mid_price
    log_return = pl.col("mid_price").log().diff(1).fill_null(0)

    # Calculate Realized Volatility
    realized_vol_10s = log_return.rolling_std(window_size=10).fill_null(0)
    realized_vol_60s = log_return.rolling_std(window_size=60).fill_null(0)
    realized_vol_300s = log_return.rolling_std(window_size=300).fill_null(0)

    # Total volume imbalance across 5 levels
    total_bid_vol = sum(pl.col(f"bid_volume{i}") for i in range(1, 6))
    total_ask_vol = sum(pl.col(f"ask_volume{i}") for i in range(1, 6))
    total_imbalance = (
        (total_bid_vol - total_ask_vol) /
        (total_bid_vol + total_ask_vol)
    )

    df = df.with_columns(
        wap1=wap1,
        obi1=obi1,
        spread1=spread1,
        log_return=log_return,
        realized_vol_10s=realized_vol_10s,
        realized_vol_60s=realized_vol_60s,
        realized_vol_300s=realized_vol_300s,
        total_imbalance=total_imbalance,
    )
    
    # Create lag features
    feature_cols = ["wap1", "obi1", "spread1", "realized_vol_10s"]
    lag_expressions = []
    for col in feature_cols:
        for lag in [1, 5, 10, 30]: # 1s, 5s, 10s, 30s ago
            lag_expressions.append(
                pl.col(col).shift(lag).alias(f"{col}_lag{lag}")
            )
    
    df = df.with_columns(lag_expressions)

    # Select final features
    feature_cols_to_keep = [
        "timestamp",
        "wap1", "obi1", "spread1", "realized_vol_10s", "realized_vol_60s", 
        "realized_vol_300s", "total_imbalance"
    ] + [expr.meta.output_name() for expr in lag_expressions]

    # Rename all features with the asset prefix
    if asset_name:
        final_cols = ["timestamp"] + [
            pl.col(c).alias(f"{prefix}{c}") for c in feature_cols_to_keep if c != "timestamp"
        ]
        return df.select(final_cols)
    else:
        # For the target asset, no prefix is needed
        return df.select(feature_cols_to_keep)


def get_cross_asset_names(directory: Path, target_asset: str) -> List[str]:
    """Scans a directory for CSV files to use as cross-assets."""
    all_files = [f.stem for f in directory.glob("*.csv")]
    return [asset for asset in all_files if asset != target_asset]


def load_and_engineer_data(
    directory: Path, 
    is_test: bool = False
) -> Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    """
    Loads and merges all required data for train or test.
    """
    logger.info(f"Loading data from {directory}...")
    
    # --- 1. Load target asset (ETH) ---
    target_schema = Config.get_schema_overrides()
    if is_test:
        if 'label' in target_schema:
            del target_schema['label']
            
    target_path = directory / f"{Config.TARGET_ASSET}.csv"
    if not target_path.exists():
        logger.error(f"Target asset file not found: {target_path}")
        raise FileNotFoundError(f"Target asset file not found: {target_path}")

    target_df = pl.read_csv(
        target_path,
        schema_overrides=target_schema,
        try_parse_dates=True
    )
    
    # --- 2. Engineer features for target asset ---
    logger.info(f"Engineering features for {Config.TARGET_ASSET}...")
    features_df = create_orderbook_features(target_df, asset_name="") 
    
    if not is_test:
        features_df = features_df.with_columns(label=target_df[Config.TARGET_COL])
        
    original_test_timestamps = None
    if is_test:
        original_test_timestamps = target_df.select("timestamp")

    del target_df
    gc.collect()

    # --- 3. Load and merge cross-asset features ---
    cross_schema = Config.get_schema_overrides()
    if 'label' in cross_schema:
        del cross_schema['label']

    if Config.CROSS_ASSETS is None:
        cross_assets_to_load = get_cross_asset_names(directory, Config.TARGET_ASSET)
    else:
        cross_assets_to_load = Config.CROSS_ASSETS

    logger.info(f"Loading cross-assets: {cross_assets_to_load}")
    
    for asset in cross_assets_to_load:
        logger.info(f"Loading and engineering features for cross-asset {asset}...")
        asset_path = directory / f"{asset}.csv"
        if not asset_path.exists():
            logger.warning(f"File for asset {asset} not found at {asset_path}. Skipping.")
            continue
        
        try:
            cross_df = pl.read_csv(
                asset_path,
                schema_overrides=cross_schema,
                try_parse_dates=True
            )
            cross_features_df = create_orderbook_features(cross_df, asset_name=asset)
            features_df = features_df.join(cross_features_df, on="timestamp", how="left")
            del cross_df, cross_features_df
            gc.collect()
        except Exception as e:
            logger.warning(f"Could not load or process {asset}. Skipping. Error: {e}")

    # --- 4. Handle NaNs ---
    # NaNs arise from rolling windows, lags, and left joins
    if is_test:
        # For test, we must predict on every row. Forward fill and then backfill.
        features_df = features_df.fill_null(strategy="forward").fill_null(0)
    else:
        # For train, it's safer to drop rows with NaNs to avoid training on
        # incomplete/imputed data from the start of the series.
        features_df = features_df.drop_nulls()

    logger.info("Data loading and feature engineering complete.")
    return features_df, original_test_timestamps


# --- 4. Model Training & Evaluation ---
def train_model(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray
) -> lgb.LGBMRegressor:
    """Trains the LightGBM model with validation."""
    
    logger.info("--- Training LightGBM model ---")
    model = lgb.LGBMRegressor(**Config.LGBM_PARAMS)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        **Config.LGBM_FIT_PARAMS
    )
    return model


def evaluate_model(
    model: lgb.LGBMRegressor, 
    X_val: np.ndarray, 
    y_val: np.ndarray, 
    features: List[str]
) -> None:
    """Calculates metrics and prints feature importance."""
    
    logger.info("--- Evaluating model on validation set ---")
    val_preds = model.predict(X_val)
    
    corr, _ = pearsonr(y_val, val_preds)
    mae = np.mean(np.abs(y_val - val_preds))
    
    logger.info(f"Validation Pearson Correlation: {corr:.6f}")
    logger.info(f"Validation MAE: {mae:.6f}")

    logger.info("--- Analyzing Feature Importance ---")
    try:
        importances = model.feature_importances_
        feature_importance_df = pl.DataFrame({
            'feature': features,
            'importance': importances
        }).sort(by='importance', descending=True)
        
        logger.info("Top 30 most important features:")
        logger.info("\n" + str(feature_importance_df.head(30)))
    except Exception as e:
        logger.warning(f"Could not print feature importance: {e}")


# --- 5. Main Execution ---
def main():
    """Main function to run the full training and prediction pipeline."""
    
    # --- 2. Load and prepare training data ---
    train_df = load_and_engineer_data(Config.TRAIN_DIR, is_test=False)[0]

    features = [col for col in train_df.columns if col not in ["timestamp", Config.TARGET_COL]]
    
    # Time-based split
    split_index = int(len(train_df) * Config.VALIDATION_SPLIT_RATIO)

    X_train_df = train_df.slice(0, split_index)[features]
    y_train_df = train_df.slice(0, split_index)[Config.TARGET_COL]
    X_val_df = train_df.slice(split_index)[features]
    y_val_df = train_df.slice(split_index)[Config.TARGET_COL]

    logger.info(f"Training data shape: {X_train_df.shape}")
    logger.info(f"Validation data shape: {X_val_df.shape}")

    del train_df
    gc.collect()

    # Convert to NumPy for LightGBM
    X_train = X_train_df.to_numpy()
    y_train = y_train_df.to_numpy()
    X_val = X_val_df.to_numpy()
    y_val = y_val_df.to_numpy()
    
    del X_train_df, y_train_df, X_val_df, y_val_df
    gc.collect()

    # --- 3. Train Model ---
    model = train_model(X_train, y_train, X_val, y_val)

    # --- 4. Evaluate Model ---
    evaluate_model(model, X_val, y_val, features)

    logger.info(f"--- Saving trained model to {Config.MODEL_OUTPUT_PATH} ---")
    joblib.dump(model, Config.MODEL_OUTPUT_PATH)
    logger.info("Model successfully saved.")

    # --- 5. Generate Predictions ---
    logger.info("--- Generating predictions on test set ---")
    del X_train, y_train, X_val, y_val
    gc.collect()

    test_df, original_test_timestamps = load_and_engineer_data(
        Config.TEST_DIR, 
        is_test=True
    )
    
    # Ensure test features match train features
    X_test_df = test_df[features]
    X_test = X_test_df.to_numpy()
    
    del test_df, X_test_df
    gc.collect()
    
    test_preds = model.predict(X_test)

    logger.info("Creating submission file...")
    submission_df = original_test_timestamps.with_columns( # type: ignore
        pl.Series(name=Config.TARGET_COL, values=test_preds)
    )
    submission_df.write_csv(Config.SUBMISSION_PATH)

    logger.info(f"--- Forecast complete. Submission file '{Config.SUBMISSION_PATH}' created. ---")


if __name__ == "__main__":
    main()