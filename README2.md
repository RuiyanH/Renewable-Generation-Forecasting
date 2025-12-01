# Probabilistic Load and Renewable Generation Forecasting

## Overview

This project builds and evaluates point and probabilistic forecasts for regional electricity load (and optionally solar/wind generation) using open grid and weather data. The goal is to move beyond single “best guess” predictions and instead produce **calibrated predictive distributions** that can inform planning decisions in high-renewables power systems.

I compare simple baselines, classical statistical models, and machine-learning methods, and then wrap them in **quantile regression** and **conformal prediction** frameworks to obtain uncertainty intervals. Models are evaluated on both accuracy and calibration, including their behavior during extreme demand events.

## Data

- **Region:** e.g., `NYISO – [Zone Name]` or `PJM system load`
- **Time span:** e.g., `2018–2023`, hourly data
- **Targets:**
  - `load_mw`: system load in MW
  - *(optional)* `solar_mw`, `wind_mw`: renewable generation
- **Predictors:**
  - Lagged load and rolling statistics (last hour, last day, last week)
  - Calendar features (hour of day, day of week, month, weekend, holiday)
  - Weather features (temperature, etc.)

All data used here is publicly available; see `data/README.md` for download and preprocessing instructions.

## Methods

### Point Forecasting

- Naive baselines:
  - Persistence (next hour = last hour)
  - Same hour last week
- Classical models:
  - Linear / ridge regression with engineered features
- Machine-learning models:
  - Random Forest
  - Gradient Boosted Trees

### Probabilistic Forecasting

1. **Quantile Regression**
   - Gradient boosting models trained to predict conditional quantiles
   - Nominal quantiles: e.g., 10%, 50%, 90%
   - Intervals: [q_0.1, q_0.9] around the median

2. **Conformal Prediction**
   - Point models calibrated on a validation set to construct distribution-free prediction intervals
   - Control of finite-sample coverage without strong distributional assumptions

## Evaluation

- **Point metrics:** MAE, RMSE
- **Probabilistic metrics:**
  - Empirical coverage at nominal levels (e.g., 80%, 90%)
  - Average interval width (sharpness)
  - Pinball loss for quantile forecasts (optional)

I also examine model performance on:
- Typical days vs. extreme days (e.g., heat waves, peak load events)
- Short-horizon (1-hour-ahead) vs longer-horizon (24-hours-ahead) forecasts

Key figures are saved in `figures/` and replicated in the notebooks.

## Repository Structure

- `notebooks/`: step-by-step analyses (exploration, features, models, evaluation)
- `src/`: reusable code for loading data, feature engineering, modeling, and evaluation
- `data/`: instructions and scripts for obtaining and preprocessing data
- `figures/`: exported plots used in writeups

## Motivation

High-renewables power systems require not only accurate point forecasts of demand and generation, but also **well-calibrated uncertainty estimates**, especially around rare but system-critical events. This project explores practical ways to build such probabilistic forecasts from open data using tools from statistics and machine learning.

