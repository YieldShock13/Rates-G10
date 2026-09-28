# SA Rates RV Monitor

Streamlit dashboard for South African rates relative-value analysis using RBond data.

## Metrics
- 1Y, 5Y and 10Y constant-maturity SAGB spreads versus ZARONIA
- 5s1s, 10s1s and 10s5s curve spreads
- 60-observation spread change
- 252-observation z-score
- Historical percentile
- Rolling 252-observation AR(1) persistence
- Implied half-life where the AR(1) estimate is sufficiently below the unit-root boundary

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
