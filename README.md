# Exoplanet Explorer (NASA Space Apps)

Streamlit app for exploring KOI/K2/TOI catalogs and a binary ML model (CONFIRMED vs FALSE POSITIVE).
- `streamlit_app.py` — main app
- `data/` — raw catalogs (ignored by git), add small samples to `data/samples/`
- `models/` — saved scaler/model artifacts (ignored by git)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
