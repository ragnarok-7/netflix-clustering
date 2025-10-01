# scripts/utils.py
"""
Utility helpers for Netflix clustering scripts.
Safe, defensive functions: parse genres, parse durations, ensure directories, logging.
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd

def setup_logger(name=__name__, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s", "%Y-%m-%d %H:%M:%S")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger

logger = setup_logger("netflix_scripts", level=logging.INFO)

def safe_load_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path.resolve()}")
    df = pd.read_csv(path)
    return df

def ensure_outputs_dir(path="outputs"):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def ensure_list(x):
    """Normalize genre-like inputs into a Python list of strings."""
    if pd.isna(x) or x == "":
        return []
    if isinstance(x, list):
        return x
    s = str(x)
    # Try to handle stringified lists like "['Action', 'Drama']"
    if s.startswith("[") and s.endswith("]"):
        try:
            s2 = s.strip("[]")
            parts = [p.strip().strip("'\"") for p in s2.split(",") if p.strip()]
            return parts
        except Exception:
            pass
    # Fallback: split by comma
    return [p.strip() for p in s.split(",") if p.strip()]

def parse_duration_to_num(x):
    """Parse duration like '90 min' or '2 Seasons' into integer; return np.nan on unknown."""
    if pd.isna(x):
        return np.nan
    s = str(x).lower().strip()
    # e.g., "90 min"
    if "min" in s:
        try:
            return int(s.split()[0])
        except Exception:
            return np.nan
    # e.g., "1 Season" or "2 Seasons"
    if "season" in s:
        try:
            return int(s.split()[0])
        except Exception:
            return np.nan
    return np.nan
