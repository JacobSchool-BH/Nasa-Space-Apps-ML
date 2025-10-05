# src/constants.py
import os

# Where your local CSVs live (put your downloaded files here)
RAW_DATA_DIR = "data/raw"

# Default filenames (change if your file names differ)
KOI_CSV = os.path.join(RAW_DATA_DIR, "koi_cumulative.csv")  # Kepler KOI cumulative
K2_CSV  = os.path.join(RAW_DATA_DIR, "k2pandc.csv")         # K2 planets & candidates
TOI_CSV = os.path.join(RAW_DATA_DIR, "toi.csv")             # TESS TOI

# Unified schema we aim to produce after normalization
UNIFIED_COLUMNS = [
    "period", "duration", "depth", "prad",
    "impact", "snr", "steff", "slogg", "srad",
    "disposition", "mission"
]

