import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin


def compute_absolute_residuals(model: RegressorMixin, X_cal: pd.DataFrame, y_cal: pd.Series) -> np.ndarray:
	"""
	Compute absolute residuals |y - y_hat| on a calibration set.
	"""
	y_pred = model.predict(X_cal)
	# Ensure numpy array for quantile computations
	return np.abs(np.asarray(y_cal) - np.asarray(y_pred))


def conformal_interval(
	model: RegressorMixin,
	X: pd.DataFrame,
	residuals: np.ndarray,
	alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Construct conformal prediction intervals around model predictions.
	Uses the (1 - alpha) empirical quantile of absolute residuals from a calibration set.
	Returns (lower, upper) arrays aligned with rows of X.
	"""
	if not isinstance(residuals, np.ndarray):
		residuals = np.asarray(residuals)
	if residuals.ndim != 1:
		residuals = residuals.ravel()
	# Empirical quantile
	q = float(np.quantile(residuals, 1.0 - alpha))
	y_pred = np.asarray(model.predict(X))
	lower = y_pred - q
	upper = y_pred + q
	return lower, upper


