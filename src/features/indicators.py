import pandas as pd
import numpy as np


def add_sma(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    windows = windows or [10, 20, 50, 200]
    for w in windows:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
    return df


def add_ema(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    windows = windows or [10, 20, 50]
    for w in windows:
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    df[f"RSI_{period}_ma"] = df[f"RSI_{period}"].rolling(3).mean()
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    sma = df["Close"].rolling(window=period).mean()
    s = df["Close"].rolling(window=period).std()
    df["BB_middle"] = sma
    df["BB_upper"] = sma + std * s
    df["BB_lower"] = sma - std * s
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
    df["BB_position"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"ATR_{period}"] = tr.rolling(window=period).mean()
    df[f"ATR_pct"] = df[f"ATR_{period}"] / df["Close"] * 100
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["OBV"] = obv
    df["OBV_SMA"] = obv.rolling(20).mean()
    df["OBV_ratio"] = obv / obv.rolling(20).mean()
    return df


def add_roc(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    df[f"ROC_{period}"] = df["Close"].pct_change(periods=period) * 100
    df[f"ROC_{period}_ma"] = df[f"ROC_{period}"].rolling(3).mean()
    return df


def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    highest = df["High"].rolling(window=period).max()
    lowest = df["Low"].rolling(window=period).min()
    df["Williams_%R"] = -100 * (highest - df["Close"]) / (highest - lowest)
    return df


def add_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    lowest = df["Low"].rolling(window=k).min()
    highest = df["High"].rolling(window=k).max()
    df["Stoch_%K"] = (df["Close"] - lowest) / (highest - lowest) * 100
    df["Stoch_%D"] = df["Stoch_%K"].rolling(window=d).mean()
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Price_Change"] = df["Close"].pct_change()
    df["Price_Change_5"] = df["Close"].pct_change(5)
    df["Price_Change_21"] = df["Close"].pct_change(21)
    df["High_Low_Ratio"] = df["High"] / df["Low"]
    df["Close_Open_Ratio"] = df["Close"] / df["Open"]
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_Change_5"] = df["Volume"].pct_change(5)
    return df


def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    df["Regime_SMA"] = df["Close"] / df["SMA_200"]
    df["Regime_Volatility"] = df["Price_Change"].rolling(21).std() * np.sqrt(252)
    return df


def add_all_indicators(df: pd.DataFrame, config=None) -> pd.DataFrame:
    df = df.copy()
    if config:
        add_sma(df, config.sma_windows)
        add_ema(df, config.ema_windows)
        add_rsi(df, config.rsi_period)
        add_macd(df, config.macd_fast, config.macd_slow, config.macd_signal)
        add_bollinger_bands(df, config.bb_period, config.bb_std)
        add_atr(df, config.atr_period)
        add_obv(df)
        add_roc(df, config.roc_period)
        add_williams_r(df, config.williams_period)
        add_stochastic(df)
    else:
        add_sma(df)
        add_ema(df)
        add_rsi(df)
        add_macd(df)
        add_bollinger_bands(df)
        add_atr(df)
        add_obv(df)
        add_roc(df)
        add_williams_r(df)
        add_stochastic(df)
    add_momentum_features(df)
    add_market_regime(df)
    return df
