from __future__ import annotations

from typing import Dict, Tuple, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae_rmse(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
	"""
	Return (MAE, RMSE).
	"""
	mae = mean_absolute_error(y_true, y_pred)
	rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
	return float(mae), rmse


def summarize_metrics(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
	"""
	Convert a nested metrics dict to a DataFrame for easy display.
	"""
	return pd.DataFrame(metrics).T.sort_values("MAE")


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
	"""
	Compute MAE and RMSE for regression.
	"""
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)
	mae = float(np.mean(np.abs(y_true - y_pred)))
	rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
	return {"MAE": mae, "RMSE": rmse}


def interval_coverage(y_true, lower, upper) -> float:
	"""
	Empirical coverage of prediction intervals.
	"""
	y_true = np.asarray(y_true)
	lower = np.asarray(lower)
	upper = np.asarray(upper)
	inside = (y_true >= lower) & (y_true <= upper)
	return float(inside.mean())


def interval_width(lower, upper) -> float:
	"""
	Average width of prediction intervals.
	"""
	lower = np.asarray(lower)
	upper = np.asarray(upper)
	return float(np.mean(upper - lower))


def pinball_loss(y_true, y_pred, quantile: float) -> float:
	"""
	Pinball loss for a given quantile.
	"""
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)
	diff = y_true - y_pred
	return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def quantile_calibration(
	y_true: np.ndarray,
	q_pred_dict: Mapping[float, np.ndarray],
) -> pd.DataFrame:
	"""
	Compute empirical coverage for multiple quantile levels.
	Returns DataFrame with columns ['quantile', 'empirical_cdf'].
	"""
	y_true = np.asarray(y_true)
	rows = []
	for q, q_pred in sorted(q_pred_dict.items()):
		q_pred = np.asarray(q_pred)
		empirical = float(np.mean(y_true <= q_pred))
		rows.append({"quantile": float(q), "empirical_cdf": empirical})
	return pd.DataFrame(rows)


def plot_quantile_calibration(calib_df: pd.DataFrame) -> None:
	"""
	Plot quantile calibration curve: nominal q vs empirical CDF.
	"""
	plt.figure(figsize=(5, 5))
	plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
	plt.plot(calib_df["quantile"], calib_df["empirical_cdf"], marker="o", label="Model")
	plt.xlabel("Nominal quantile")
	plt.ylabel("Empirical CDF")
	plt.title("Quantile calibration")
	plt.legend()
	plt.tight_layout()
	plt.show()


