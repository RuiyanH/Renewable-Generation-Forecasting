from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import holidays


def add_calendar_features(df: pd.DataFrame, country: str = "DE") -> pd.DataFrame:
	"""
	Add calendar features: hour, day_of_week, month, is_weekend, is_holiday.
	"""
	if not isinstance(df.index, pd.DatetimeIndex):
		raise ValueError("DataFrame must be indexed by DatetimeIndex for calendar features.")

	out = df.copy()
	out["hour"] = out.index.hour
	out["day_of_week"] = out.index.dayofweek
	out["month"] = out.index.month
	out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

	year_span = range(out.index.min().year, out.index.max().year + 1)
	calendar = holidays.country_holidays(country=country, years=year_span)
	out["is_holiday"] = out.index.normalize().isin(calendar).astype(int)
	return out


def add_lagged_load_features(
	df: pd.DataFrame,
	target_col: str = "load_mw",
	lag_hours: Iterable[int] = (1, 24, 168),
	rolling_windows: Iterable[int] = (24, 168),
) -> pd.DataFrame:
	"""
	Add lagged target values and non-leaky rolling statistics.
	- Lags: t-1, t-24, t-168 by default
	- Rolling means with a 1 hour lag offset: mean over last 24/168 hours excluding current hour
	"""
	out = df.copy()
	if target_col not in out.columns:
		raise ValueError(f"Target column '{target_col}' not in DataFrame.")

	for h in lag_hours:
		out[f"{target_col}_lag_{h}h"] = out[target_col].shift(h)

	for w in rolling_windows:
		out[f"{target_col}_rollmean_{w}h"] = out[target_col].shift(1).rolling(window=w, min_periods=max(1, w // 2)).mean()
		out[f"{target_col}_rollstd_{w}h"] = out[target_col].shift(1).rolling(window=w, min_periods=max(1, w // 2)).std()
	return out


def select_feature_columns(
	df: pd.DataFrame,
	target_col: str = "load_mw",
	extra_cols: Iterable[str] = ("temperature_c", "solar_mw", "wind_mw"),
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
	"""
	Split DataFrame into X and y with a reasonable default feature set.
	Returns (X, y, feature_names).
	"""
	candidate_features: List[str] = []
	# Engineered features
	for col in df.columns:
		if col.startswith(f"{target_col}_lag_") or col.startswith(f"{target_col}_roll"):
			candidate_features.append(col)
	# Calendar
	for cal in ("hour", "day_of_week", "month", "is_weekend", "is_holiday"):
		if cal in df.columns:
			candidate_features.append(cal)
	# Add optional raw predictors if present
	for name in extra_cols:
		if name in df.columns:
			candidate_features.append(name)

	candidate_features = sorted(set(candidate_features))
	X = df[candidate_features].copy()
	y = df[target_col].copy()
	return X, y, candidate_features


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Add calendar and cyclical time features derived from the DatetimeIndex:
	- hour_sin, hour_cos (cyclical encoding for hour of day)
	- day_of_week (0=Mon ... 6=Sun)
	- is_weekend (Sat/Sun)
	- month (1..12)
	"""
	if not isinstance(df.index, pd.DatetimeIndex):
		raise ValueError("DataFrame must be indexed by DatetimeIndex.")
	out = df.copy()
	dt = out.index
	out["day_of_week"] = dt.dayofweek
	out["month"] = dt.month
	out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
	# Cyclical encoding for hour
	hour = dt.hour
	out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
	out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
	return out


def add_lagged_features(df: pd.DataFrame, lags: Iterable[int] = (1, 2, 24, 168)) -> pd.DataFrame:
	"""
	Add lag features for ['load', 'solar', 'wind'] at the provided horizons.
	"""
	required = ["load", "solar", "wind"]
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")
	out = df.copy()
	for col in required:
		for lag in lags:
			out[f"{col}_lag_{lag}"] = out[col].shift(lag)
	return out


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Add non-leaky rolling means/stds:
	- load: 24h mean/std, 168h mean/std
	- solar: 24h mean/std
	- wind: 24h mean/std
	Uses shift(1) before rolling to avoid leakage of the current hour.
	"""
	out = df.copy()
	# Load
	out["load_roll_mean_24"] = out["load"].shift(1).rolling(24, min_periods=12).mean()
	out["load_roll_std_24"] = out["load"].shift(1).rolling(24, min_periods=12).std()
	out["load_roll_mean_168"] = out["load"].shift(1).rolling(168, min_periods=84).mean()
	out["load_roll_std_168"] = out["load"].shift(1).rolling(168, min_periods=84).std()
	# Solar
	out["solar_roll_mean_24"] = out["solar"].shift(1).rolling(24, min_periods=12).mean()
	out["solar_roll_std_24"] = out["solar"].shift(1).rolling(24, min_periods=12).std()
	# Wind
	out["wind_roll_mean_24"] = out["wind"].shift(1).rolling(24, min_periods=12).mean()
	out["wind_roll_std_24"] = out["wind"].shift(1).rolling(24, min_periods=12).std()
	return out


def make_supervised(df: pd.DataFrame, horizon: int = 1, target_col: str = "load") -> Tuple[pd.DataFrame, pd.Series]:
	"""
	Turn a time series DataFrame with engineered features into (X, y) for a given forecast horizon.
	Aligns so that row at time t in X predicts target at time t + horizon.
	"""
	if target_col not in df.columns:
		raise ValueError(f"Target column '{target_col}' not found.")
	out = df.copy()
	out["target"] = out[target_col].shift(-horizon)
	out = out.dropna()
	y = out["target"]
	X = out.drop(columns=["target"])
	return X, y


def make_features(raw: pd.DataFrame, horizon: int = 1, target_col: str = "load") -> Tuple[pd.DataFrame, pd.Series]:
	"""
	Feature pipeline for DataFrames with columns ['load', 'solar', 'wind'] indexed by hour.
	Steps:
	- add_time_features
	- add_lagged_features for lags 1,2,24,168 (for load/solar/wind)
	- add_rolling_features: load (24h/168h mean/std), solar (24h mean/std), wind (24h mean/std)
	- drop rows with NaNs introduced by shifting/rolling
	- make_supervised to return (X, y) for the specified horizon
	"""
	df = raw.copy()
	df = add_time_features(df)
	df = add_lagged_features(df, lags=(1, 2, 24, 168))
	df = add_rolling_features(df)
	df = df.dropna()
	return make_supervised(df, horizon=horizon, target_col=target_col)

