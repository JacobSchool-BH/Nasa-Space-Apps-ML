# src/datasets.py
import os
import pandas as pd
import yaml

from src.constants import KOI_CSV, K2_CSV, TOI_CSV, UNIFIED_COLUMNS

# --- put near the top of src/datasets.py ---
import io

import io
import pandas as pd

def _read_csv_smart(path: str, expected_cols: list[str]) -> pd.DataFrame:
    """
    Robust CSV loader:
    1) Try the default fast path (C engine).
    2) If columns not found, try Python engine with sep sniffing (no low_memory).
    3) If still failing, detect the header line containing expected cols and re-read from there.
    """
    # --- 1) Fast path (C engine)
    try:
        df = pd.read_csv(path)  # default engine='c'
        if all(c in df.columns for c in expected_cols):
            return df
    except Exception:
        pass

    # --- 2) Python engine with sep sniffing (NO low_memory here)
    try:
        df = pd.read_csv(path, engine="python", sep=None, on_bad_lines="skip")
        if all(c in df.columns for c in expected_cols):
            return df
    except Exception:
        pass

    # --- 3) Scan first ~100 lines to find the real header row
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines[:200]):  # scan a bit more just in case
        # crude comma split is enough to find candidate header line
        cols = [c.strip().lower() for c in line.strip().split(",")]
        if all(ec.lower() in cols for ec in expected_cols):
            header_idx = i
            break

    if header_idx is None:
        # final fallback: try python engine once more on the whole file
        df = pd.read_csv(path, engine="python", sep=None, on_bad_lines="skip")
        if all(c in df.columns for c in expected_cols):
            return df
        raise ValueError(
            f"Could not locate a header containing {expected_cols} in {path}. "
            "The file may be an HTML error page or a different export."
        )

    # Re-read from detected header downwards (no low_memory with python engine)
    text = "".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(text), engine="python", sep=None, on_bad_lines="skip")
    return df



def _maybe_load_yaml(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}

def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_koi(path: str) -> pd.DataFrame:
    # use the smart reader you added earlier (or just pd.read_csv if you replaced the file)
    df = pd.read_csv(path)  # if you’re using the cleaned file

    # SNR fallback: some cumulative exports use koi_model_snr
    if "koi_snr" not in df.columns and "koi_model_snr" in df.columns:
        df["koi_snr"] = df["koi_model_snr"]

    mapping = {
        "koi_period": "period",
        "koi_duration": "duration",
        "koi_depth": "depth",
        "koi_prad": "prad",
        "koi_impact": "impact",
        "koi_snr": "snr",
        "koi_steff": "steff",
        "koi_slogg": "slogg",
        "koi_srad": "srad",
        "koi_disposition": "disposition",
    }
    have = [c for c in mapping if c in df.columns]
    df = df[have].rename(columns={c: mapping[c] for c in have})
    df["mission"] = "kepler"
    return df



def load_k2(path: str) -> pd.DataFrame:
    """K2 planets & candidates → unify column names and dispositions."""
    df = pd.read_csv(path)
    mapping = {
        "pl_orbper": "period",
        "t14": "duration",
        "tran_depth": "depth",
        "pl_rade": "prad",
        "impact": "impact",
        "snr": "snr",
        "st_teff": "steff",
        "st_logg": "slogg",
        "st_rad": "srad",
        "disposition": "disposition",
    }
    have = [c for c in mapping if c in df.columns]
    df = df[have].rename(columns={c: mapping[c] for c in have})
    df["mission"] = "k2"
    # Normalize dispositions
    df["disposition"] = (
        df["disposition"]
        .astype(str)
        .str.upper()
        .map({
            "CONFIRMED": "CONFIRMED",
            "CANDIDATE": "CANDIDATE",
            "CP": "CANDIDATE",
            "PC": "CANDIDATE",
            "FALSE POSITIVE": "FALSE POSITIVE",
            "FP": "FALSE POSITIVE",
        })
        .fillna("CANDIDATE")
    )
    return df

def load_toi(path: str) -> pd.DataFrame:
    """TESS TOI → unify column names and dispositions."""
    df = pd.read_csv(path)
    mapping = {
        "period": "period",
        "duration": "duration",
        "depth": "depth",
        "planet_radius": "prad",
        "impact": "impact",
        "snr": "snr",
        "st_teff": "steff",
        "st_logg": "slogg",
        "st_rad": "srad",
        "disposition": "disposition",
    }
    have = [c for c in mapping if c in df.columns]
    df = df[have].rename(columns={c: mapping[c] for c in have})
    df["mission"] = "tess"
    df["disposition"] = (
        df["disposition"]
        .astype(str)
        .str.upper()
        .map({
            "CONFIRMED": "CONFIRMED",
            "CANDIDATE": "CANDIDATE",
            "CP": "CANDIDATE",
            "PC": "CANDIDATE",
            "FALSE POSITIVE": "FALSE POSITIVE",
            "FP": "FALSE POSITIVE",
        })
        .fillna("CANDIDATE")
    )
    return df

def load_all(sources=("koi",), config_yaml="config/data_paths.yaml") -> pd.DataFrame:
    """Load 1+ missions from local CSVs and merge into a unified DataFrame."""
    cfg = _maybe_load_yaml(config_yaml)
    out = []
    if "koi" in sources:
        out.append(load_koi(cfg.get("koi_csv", KOI_CSV)))
    if "k2" in sources:
        out.append(load_k2(cfg.get("k2_csv", K2_CSV)))
    if "toi" in sources:
        out.append(load_toi(cfg.get("toi_csv", TOI_CSV)))

    if not out:
        raise ValueError("No sources loaded. Check sources or paths.")

    df = pd.concat(out, ignore_index=True, sort=False)

    # keep only unified columns that exist
    keep = [c for c in UNIFIED_COLUMNS if c in df.columns]
    df = df[keep]

    num_cols = ["period","duration","depth","prad","impact","snr","steff","slogg","srad"]
    df = _coerce_numeric(df, [c for c in num_cols if c in df.columns])

    df = df.dropna(subset=["disposition"])  # must have labels
    return df
