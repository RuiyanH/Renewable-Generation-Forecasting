## Germany hourly load forecasting with probabilistic uncertainty

### Overview
High-renewables power systems require not only accurate point forecasts of demand and generation, but also well-calibrated uncertainty estimates, especially around rare but critical events. This project explores practical ways to build such probabilistic forecasts from open data using tools from statistics and machine learning.

Specifically, I choose to predict Germany's hourly electricity load and solar/wind generation using open grid and weather data. I compare linear models, tree-based methods, and neural networks, then wrap them in quantile and conformal prediction frameworks to produce full predictive distributions. I evaluate models by accuracy and calibration, especially on extreme events like peak load days and low-renewable periods, to illustrate how probabilistic forecasts can better support dispatch, storage, and investment decisions in high-renewables grids.

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

All data used here is publicly available; see `data/README.md` for download and preprocessing instructions.

### Methods
I use public hourly time-series data for Germany from Open Power System Data, including system load and aggregate solar and wind generation. I construct supervised learning datasets by engineering lagged and rolling features (1h, 24h, 168h lags; 24h and 168h rolling statistics), calendar features (hour-of-day, day-of-week, month, weekend), and optional net-load features. I compare naive baselines (persistence and same-hour-last-week) to tree-based point models (gradient-boosted regression trees).

To obtain probabilistic forecasts, I implement two approaches. First, I train separate gradient-boosting models with quantile loss to directly predict conditional quantiles (e.g., 10th, 50th, 90th percentiles), which define prediction intervals. 

Next, I apply distribution-free conformal prediction on top of the best point model. I reserve a calibration set, compute absolute residuals, and use their empirical quantiles to construct finite-sample-valid prediction intervals with target coverage (e.g., 90%). Models are evaluated out-of-sample using MAE and RMSE for point predictions, and empirical coverage, average interval width, pinball loss, and quantile calibration plots for probabilistic predictions.

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
1) Create and activate a conda environment (recommended):

    ```bash
    conda create -n forecast -y python=3.11
    conda activate forecast
    ```

2) Install dependencies:
   
   ```bash
    pip install -r requirements.txt
    ```

3) Download the OPSD dataset (time_series_60min_singleindex.csv):
   - Website: Open Power System Data (Time series) — `https://data.open-power-system-data.org/time_series/`
   - File: time_series_60min_singleindex.csv (Germany columns included: DE_load_actual_entsoe_transparency, DE_solar_generation_actual, DE_wind_generation_actual)
   - Place the file at `data/time_series_60min_singleindex.csv`
   - Example via curl:
     curl -L -o data/time_series_60min_singleindex.csv "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv"

4) Open the notebooks in notebooks/. Run them top-to-bottom.
   - If paths differ, update the DATA_PATH/RAW_PATH at the top of the notebooks.

### Data notes
- Loader: `src.data.load_opsd_germany(path)` parses `utc_timestamp` and returns columns `['load', 'solar', 'wind']` from the DE columns.
- Feature pipeline: `src.features.make_features(df, horizon=H)` builds lags (1/2/24/168), rolling stats, and calendar features, returns (X, y).
- Large files are ignored by `.gitignore` (CSV, parquet, figures, processed data). Keep raw data locally; don’t commit it.

### Results
For 1-hour-ahead load forecasting, the gradient-boosted model reduces RMSE by approximately 64% and MAE by 63% relative to a persistence baseline, and also outperforms the same-hour-last-week heuristic. Direct quantile regression yields 80% prediction intervals whose empirical coverage is close to the nominal level (about 80%) with relatively narrow average widths, while conformal prediction achieves roughly (target-level) coverage for 90% intervals at the cost of somewhat wider bands.

Visual inspection over several high-load weeks shows that both approaches capture daily load shape well, but conformal intervals are more conservative around sharp peaks and unusual patterns, providing additional protection against undercoverage in extreme conditions. The quantile calibration plots lie close to the ideal 45° line, indicating that the predicted quantiles are reasonably well calibrated. Overall, the experiment illustrates how relatively simple models, combined with careful feature engineering and conformal or quantile-based methods, can provide calibrated predictive distributions rather than just point estimates—exactly the kind of uncertainty-aware forecasts needed for planning in high-renewables power systems.
