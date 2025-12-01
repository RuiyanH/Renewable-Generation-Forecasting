from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


def baseline_persistence(y: pd.Series, horizon_hours: int = 1) -> pd.Series:
	"""
	Persistence baseline: y_hat_t = y_{t-horizon}.
	"""
	return y.shift(horizon_hours)


def baseline_same_hour_last_week(y: pd.Series) -> pd.Series:
	"""
	Weekly seasonal baseline: y_hat_t = y_{t-168}.
	"""
	return y.shift(168)


@dataclass
class TrainedModels:
	linear: LinearRegression
	ridge: Ridge
	random_forest: RandomForestRegressor
	gradient_boosting: GradientBoostingRegressor


def train_point_models(X_train: pd.DataFrame, y_train: pd.Series) -> TrainedModels:
	"""
	Train a small suite of point forecasting models.
	"""
	linear = LinearRegression()
	linear.fit(X_train, y_train)

	ridge = Ridge(alpha=10.0)
	ridge.fit(X_train, y_train)

	rf = RandomForestRegressor(
		n_estimators=300,
		max_depth=None,
		min_samples_leaf=2,
		n_jobs=-1,
		random_state=42,
	)
	rf.fit(X_train, y_train)

	gbr = GradientBoostingRegressor(
		n_estimators=400,
		learning_rate=0.05,
		max_depth=3,
		subsample=0.8,
		random_state=42,
	)
	gbr.fit(X_train, y_train)

	return TrainedModels(
		linear=linear,
		ridge=ridge,
		random_forest=rf,
		gradient_boosting=gbr,
	)


def evaluate_point_models(
	models: TrainedModels, X: pd.DataFrame, y_true: pd.Series
) -> Dict[str, Dict[str, float]]:
	"""
	Compute MAE and RMSE for each trained model on provided data.
	"""
	results: Dict[str, Dict[str, float]] = {}
	preds = {
		"linear": models.linear.predict(X),
		"ridge": models.ridge.predict(X),
		"random_forest": models.random_forest.predict(X),
		"gradient_boosting": models.gradient_boosting.predict(X),
	}
	for name, y_hat in preds.items():
		mae = mean_absolute_error(y_true, y_hat)
		rmse = np.sqrt(mean_squared_error(y_true, y_hat))
		results[name] = {"MAE": float(mae), "RMSE": float(rmse)}
	return results


def train_point_model(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingRegressor:
	"""
	Train a single GradientBoostingRegressor for point forecasting.
	"""
	model = GradientBoostingRegressor(
		loss="squared_error",
		n_estimators=300,
		max_depth=3,
		learning_rate=0.05,
		random_state=42,
	)
	model.fit(X_train, y_train)
	return model


def train_quantile_models(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	quantiles: List[float] = [0.1, 0.5, 0.9],
) -> Dict[float, GradientBoostingRegressor]:
	"""
	Train one GradientBoostingRegressor per requested quantile.
	"""
	models: Dict[float, GradientBoostingRegressor] = {}
	for q in quantiles:
		m = GradientBoostingRegressor(
			loss="quantile",
			alpha=q,
			n_estimators=300,
			max_depth=3,
			learning_rate=0.05,
			random_state=42,
		)
		m.fit(X_train, y_train)
		models[q] = m
	return models


def predict_quantiles(models: Dict[float, GradientBoostingRegressor], X: pd.DataFrame) -> pd.DataFrame:
	"""
	Predict conditional quantiles for rows in X using a dict of trained models keyed by quantile.
	Returns a DataFrame with columns like 'q_0.1', 'q_0.5', 'q_0.9'.
	"""
	preds: Dict[str, np.ndarray] = {}
	for q, m in models.items():
		col_name = f"q_{q}"
		preds[col_name] = m.predict(X)
	index = X.index if hasattr(X, "index") else None
	return pd.DataFrame(preds, index=index)


