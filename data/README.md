OPSD Germany time series – download guide

Source
- Open Power System Data – Time series: `https://data.open-power-system-data.org/time_series/`
- File: `time_series_60min_singleindex.csv` (contains hourly system load and renewable generation by country)

Quick download (macOS/Linux)
- Using curl:
  curl -L -o data/time_series_60min_singleindex.csv "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv"
- Using wget:
  wget -O data/time_series_60min_singleindex.csv "https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv"

Expected schema (subset used)
- Index: `utc_timestamp` (parsed to timezone-aware DatetimeIndex)
- Germany columns consumed by this repo:
  - `DE_load_actual_entsoe_transparency` → `load`
  - `DE_solar_generation_actual` → `solar`
  - `DE_wind_generation_actual` → `wind`

Loader
- Use `src.data.load_opsd_germany("data/time_series_60min_singleindex.csv")`
  This parses `utc_timestamp`, sets it as index, and returns only `['load', 'solar', 'wind']` (MW).

Tips
- Keep the raw CSV out of git (see `.gitignore` in repo root).
- If you want to fetch a specific historical version, browse the OPSD time_series releases page and adjust the URL accordingly.



