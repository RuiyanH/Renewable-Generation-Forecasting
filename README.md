## Germany hourly load forecasting with probabilistic uncertainty

### Overview
This repository builds point and probabilistic forecasts for Germany’s hourly electricity load, with optional renewables and weather predictors. It emphasizes practical, open-data workflows and well-calibrated uncertainty around rare but system-critical events.

Probabilistic Load and Renewable Generation Forecasting for High-Renewables Power Systems
I build models that predict regional electricity load and solar/wind generation using open grid and weather data, with a focus on calibrated uncertainty rather than just point forecasts. Starting from simple baselines, I compare linear models, tree-based methods, and neural networks, then wrap them in quantile and conformal prediction frameworks to produce full predictive distributions. I evaluate models by both accuracy and calibration, especially on extreme events like peak load days and low-renewable periods, to illustrate how probabilistic forecasts can better support dispatch, storage, and investment decisions in high-renewables grids.

### Data
- Region: Germany
- Time span: 2020–2025 (hourly)
- Targets:
  - load_mw (system load)
  - optional: solar_mw, wind_mw
- Predictors:
  - Lagged load and rolling statistics (last hour, last day, last week)
  - Calendar features (hour, day-of-week, month, weekend, holiday)
  - Weather features (e.g., temperature)

All data used here is publicly available; see data/README.md for download and preprocessing instructions.

### Methods
I use public hourly time-series data for Germany from Open Power System Data, including system load and aggregate solar and wind generation. I construct supervised learning datasets by engineering lagged and rolling features (1h, 24h, 168h lags; 24h and 168h rolling statistics), calendar features (hour-of-day, day-of-week, month, weekend), and optional net-load features. I compare naive baselines (persistence and same-hour-last-week) to tree-based point models (gradient-boosted regression trees).
To obtain probabilistic forecasts, I implement two approaches. First, I train separate gradient-boosting models with quantile loss to directly predict conditional quantiles (e.g., 10th, 50th, 90th percentiles), which define prediction intervals. Second, I apply distribution-free conformal prediction on top of the best point model: I reserve a calibration set, compute absolute residuals, and use their empirical quantiles to construct finite-sample-valid prediction intervals with target coverage (e.g., 90%). Models are evaluated out-of-sample using MAE and RMSE for point predictions, and empirical coverage, average interval width, pinball loss, and quantile calibration plots for probabilistic predictions.

#### Point forecasting
- Naive baselines:
  - Persistence (next hour = last hour)
  - Same hour last week
- Classical models:
  - Linear / ridge regression with engineered features
- Machine-learning models:
  - Random Forest
  - Gradient Boosted Trees

#### Probabilistic forecasting
1) Quantile Regression
   - Gradient boosting models trained for conditional quantiles
   - Nominal quantiles: e.g., 10%, 50%, 90%
   - Intervals: [q_0.1, q_0.9] around the median
2) Conformal Prediction
   - Calibrate point models on validation data for distribution-free prediction intervals
   - Achieve finite-sample coverage control with minimal assumptions

#### Evaluation
- Point metrics: MAE, RMSE
- Probabilistic metrics:
  - Empirical coverage at nominal levels (e.g., 80%, 90%)
  - Average interval width (sharpness)
  - Pinball loss for quantile forecasts (optional)
- Analysis slices:
  - Typical vs extreme days (e.g., heat waves, peak load)
  - Short-horizon (1-hour ahead) vs longer-horizon (24-hours ahead)

I also examine model performance on:
- Typical days vs. extreme days (e.g., heat waves, peak load events)
- Short-horizon (1-hour-ahead) vs longer-horizon (24-hours-ahead) forecasts


### Repository structure
- notebooks/: step-by-step analyses (exploration, features, models, evaluation)
- src/: reusable code for data loading, feature engineering, modeling, evaluation
- data/: instructions and (optional) scripts to obtain and preprocess data
- figures/: exported plots for writeups

### Getting started
1) Create and activate a Python 3.10+ environment.
2) Install dependencies:
   pip install -r requirements.txt
3) Put your raw data files in the project root or follow data/README.md to fetch from public sources. Example file expected by notebooks: time_series_60min_singleindex.csv
4) Open the notebooks in notebooks/ and run them top-to-bottom.

### Results
For 1-hour-ahead load forecasting, the gradient-boosted model reduces RMSE by approximately X% and MAE by Y% relative to a persistence baseline, and also outperforms the same-hour-last-week heuristic. Direct quantile regression yields 80% prediction intervals whose empirical coverage is close to the nominal level (about Z%) with relatively narrow average widths, while conformal prediction achieves roughly (target-level) coverage for 90% intervals at the cost of somewhat wider bands.
Visual inspection over several high-load weeks shows that both approaches capture daily load shape well, but conformal intervals are more conservative around sharp peaks and unusual patterns, providing additional protection against undercoverage in extreme conditions. The quantile calibration plots lie close to the ideal 45° line, indicating that the predicted quantiles are reasonably well calibrated. Overall, the experiment illustrates how relatively simple models, combined with careful feature engineering and conformal or quantile-based methods, can provide calibrated predictive distributions rather than just point estimates—exactly the kind of uncertainty-aware forecasts needed for planning in high-renewables power systems.


### Notes
- The src/ package provides utilities for feature engineering and model training.
- The notebooks are intentionally simple and reproducible.
- For weather data, you can substitute any temperature source as long as it is aligned hourly and merged on timestamp.




