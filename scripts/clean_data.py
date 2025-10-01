# scripts/clean_data.py
import argparse
from scripts.utils import safe_load_csv, ensure_outputs_dir, logger, ensure_list, parse_duration_to_num
import pandas as pd
import numpy as np
from pathlib import Path

def clean_df(df):
    df = df.copy()
    df['description'] = df.get('description', "").fillna("").astype(str)
    df['listed_in'] = df.get('listed_in', "").fillna("").astype(str)
    df['genres_list'] = df['listed_in'].apply(ensure_list)
    df['duration_num'] = df.get('duration', np.nan).apply(parse_duration_to_num) if 'duration' in df.columns else np.nan
    df['release_year'] = pd.to_numeric(df.get('release_year', np.nan), errors='coerce')
    if 'title' in df.columns and 'release_year' in df.columns:
        df = df.drop_duplicates(subset=['title', 'release_year'])
    else:
        df = df.drop_duplicates()
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv")
    parser.add_argument("--out", default="outputs/cleaned_netflix.csv")
    args = parser.parse_args()
    df = safe_load_csv(args.csv)
    df_clean = clean_df(df)
    Path("outputs").mkdir(exist_ok=True)
    df_clean.to_csv(args.out, index=False)
    logger.info("Saved cleaned csv to %s", args.out)
