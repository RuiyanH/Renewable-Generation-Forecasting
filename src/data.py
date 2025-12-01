"""
Data loading and preprocessing functions.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Optional, Sequence

import pandas as pd


def _infer_datetime_column_name(columns: Sequence[str]) -> Optional[str]:
	"""
	Return the most likely datetime column name if present.
	"""
	candidate_names = (
		"datetime",
		"timestamp",
		"ts",
		"date",
		"time",
		"utc_timestamp",
		"datetime_utc",
		"ds",
	)
	lower_to_original = {str(c).lower(): c for c in columns}
	for candidate in candidate_names:
		if candidate in lower_to_original:
			return lower_to_original[candidate]
	return None


def _coerce_to_hourly_index(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
	"""
	Set index to a pandas.DatetimeIndex at hourly frequency without modifying column order unnecessarily.
	"""
	if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
		df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True, errors="coerce")
	df = df.set_index(datetime_col).sort_index()
	# Keep irregularities; downstream code may drop NA introduced by lagging/rolling
	return df


def load_time_series(path: str, table: str = "time_series", datetime_col: Optional[str] = None) -> pd.DataFrame:
	"""
	Load an hourly time series from CSV or SQLite, returning a DataFrame indexed by timestamp.

	Args:
		path: Path to a CSV or SQLite file.
		table: If loading from SQLite, the table name to read from.
		datetime_col: Optional explicit datetime column name. If provided, inference is skipped.

	Returns:
		DataFrame with DatetimeIndex, sorted ascending. Does not up/down-sample.
	"""
	if not os.path.exists(path):
		raise FileNotFoundError(f"File not found: {path}")

	lower_path = path.lower()
	if lower_path.endswith(".csv"):
		df = pd.read_csv(path)
		dt_col: Optional[str] = None
		if datetime_col is not None:
			for c in df.columns:
				if str(c).lower() == str(datetime_col).lower():
					dt_col = c
					break
		if dt_col is None:
			dt_col = _infer_datetime_column_name(df.columns)
		if dt_col is None:
			raise ValueError("Could not infer datetime column name in CSV.")
		return _coerce_to_hourly_index(df, dt_col)

	if lower_path.endswith(".sqlite") or lower_path.endswith(".db"):
		with sqlite3.connect(path) as conn:
			df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
		dt_col = None
		if datetime_col is not None:
			for c in df.columns:
				if str(c).lower() == str(datetime_col).lower():
					dt_col = c
					break
		if dt_col is None:
			dt_col = _infer_datetime_column_name(df.columns)
		if dt_col is None:
			raise ValueError("Could not infer datetime column name in SQLite table.")
		return _coerce_to_hourly_index(df, dt_col)

	raise ValueError("Unsupported file type. Provide a .csv or .sqlite file.")


def validate_columns_present(df: pd.DataFrame, required: Sequence[str]) -> None:
	"""
	Validate that all required columns exist in the DataFrame.
	"""
	missing = [col for col in required if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")


def rename_region_columns_to_standard(df: pd.DataFrame, region: str = "DE") -> pd.DataFrame:
	"""
	Add standardized columns for a given region:
	- load_mw
	- solar_mw (if available)
	- wind_mw (if available; prefers onshore if present)

	The function preserves original columns and only adds standardized aliases.
	"""
	out = df.copy()
	prefix = f"{region}_"

	# Load target
	load_candidates = [
		f"{prefix}load_actual_entsoe_transparency",
		f"{prefix}load_actual",
	]
	for cand in load_candidates:
		if cand in out.columns:
			out["load_mw"] = out[cand]
			break

	# Solar
	solar_candidates = [
		f"{prefix}solar_generation_actual",
		f"{prefix}solar_generation",
	]
	for cand in solar_candidates:
		if cand in out.columns:
			out["solar_mw"] = out[cand]
			break

	# Wind
	wind_candidates = [
		f"{prefix}wind_onshore_generation_actual",
		f"{prefix}wind_generation_actual",
		f"{prefix}wind_generation",
	]
	for cand in wind_candidates:
		if cand in out.columns:
			out["wind_mw"] = out[cand]
			break

	return out


def load_opsd_germany(path: str) -> pd.DataFrame:
	"""
	Load OPSD time series for Germany (hourly) from time_series_60min_singleindex.csv.
	Parses 'utc_timestamp' and returns a DataFrame with columns ['load', 'solar', 'wind'].
	"""
	path = os.fspath(path)
	df = pd.read_csv(path)
	if "utc_timestamp" not in df.columns:
		raise ValueError("Expected 'utc_timestamp' column in OPSD CSV.")
	df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"])
	df = df.set_index("utc_timestamp")

	cols = {
		"DE_load_actual_entsoe_transparency": "load",
		"DE_solar_generation_actual": "solar",
		"DE_wind_generation_actual": "wind",
	}
	missing = [c for c in cols.keys() if c not in df.columns]
	if missing:
		raise ValueError(f"Missing expected OPSD columns: {missing}")
	df = df[list(cols.keys())].rename(columns=cols)
	return df.sort_index()

