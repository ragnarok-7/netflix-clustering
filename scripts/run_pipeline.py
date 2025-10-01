# scripts/run_pipeline.py
"""
Full pipeline CLI:
  - parse and clean dataset
  - build features (TF-IDF + TruncatedSVD for description, MultiLabelBinarizer for genres)
  - scale and run KMeans
  - save outputs and model artifacts to outputs/
Usage:
  python scripts/run_pipeline.py --csv data/NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv --out outputs/netflix_with_clusters.csv --k 6 --sample 2000
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
from tqdm import tqdm

from scripts.utils import (
    logger,
    ensure_outputs_dir,
    safe_load_csv,
    ensure_list,
    parse_duration_to_num,
)

RANDOM_STATE = 42

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply safe cleaning: fill descriptions, parse genres, parse duration and release_year."""
    df = df.copy()
    df['description'] = df.get('description', "").fillna("").astype(str)
    df['listed_in'] = df.get('listed_in', "").fillna("").astype(str)
    df['genres_list'] = df['listed_in'].apply(ensure_list)

    # parse duration_num
    if 'duration' in df.columns:
        df['duration_num'] = df['duration'].apply(parse_duration_to_num)
    else:
        df['duration_num'] = np.nan

    if 'release_year' in df.columns:
        df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    else:
        df['release_year'] = np.nan

    # drop exact duplicates by title + release_year if both exist
    if 'title' in df.columns and 'release_year' in df.columns:
        df = df.drop_duplicates(subset=['title', 'release_year'])
    else:
        df = df.drop_duplicates()
    return df

def build_features(df: pd.DataFrame, tfidf_max_features=3000, svd_n_components=50):
    """Return combined features matrix and fitted transformers."""
    descriptions = df['description'].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_df=0.8, min_df=5, max_features=tfidf_max_features, ngram_range=(1,2))
    X_text = tfidf.fit_transform(descriptions)
    actual_svd_n = min(svd_n_components, X_text.shape[1], X_text.shape[0]-1 if X_text.shape[0]>1 else 1)
    svd = TruncatedSVD(n_components=max(1, actual_svd_n), random_state=RANDOM_STATE)
    X_text_red = svd.fit_transform(X_text)

    mlb = MultiLabelBinarizer(sparse_output=False)
    X_genres = mlb.fit_transform(df['genres_list'])

    duration = df.get('duration_num', pd.Series([np.nan]*len(df))).fillna(-1).values.reshape(-1,1)
    release_year = pd.to_numeric(df.get('release_year', pd.Series([np.nan]*len(df))), errors='coerce').fillna(-1).values.reshape(-1,1)

    X = np.hstack([X_text_red, X_genres, duration, release_year])
    return X, {'tfidf': tfidf, 'svd': svd, 'mlb': mlb}

def run_clustering(X: np.ndarray, k: int):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(Xs)
    score = silhouette_score(Xs, labels) if len(set(labels)) > 1 else float('nan')
    return labels, score, {'scaler': scaler, 'kmeans': km}

def main(csv_path, out_csv, sample, k, tfidf_max_features, svd_n_components):
    outputs = ensure_outputs_dir("outputs")
    logger.info("Loading CSV...")
    df = safe_load_csv(csv_path)

    if sample is not None and sample > 0 and len(df) > sample:
        logger.info("Sampling %d rows (seed=%d) for quick run...", sample, RANDOM_STATE)
        df = df.sample(n=sample, random_state=RANDOM_STATE).reset_index(drop=True)

    logger.info("Cleaning data...")
    df_clean = clean_df(df)
    df_clean.to_csv(outputs / "cleaned_netflix.csv", index=False)
    logger.info("Saved cleaned CSV to outputs/cleaned_netflix.csv (rows=%d)", len(df_clean))

    logger.info("Building features (TF-IDF + SVD + genres)...")
    X, fitted = build_features(df_clean, tfidf_max_features=tfidf_max_features, svd_n_components=svd_n_components)
    np.save(outputs / "X_combined.npy", X)
    joblib.dump(fitted['tfidf'], outputs / "tfidf.joblib")
    joblib.dump(fitted['svd'], outputs / "svd.joblib")
    joblib.dump(fitted['mlb'], outputs / "mlb.joblib")
    logger.info("Saved feature artifacts (tfidf, svd, mlb) and X_combined.npy")

    logger.info("Clustering with KMeans k=%d ...", k)
    labels, sil, models = run_clustering(X, k=int(k))
    df_clean['cluster'] = labels
    df_clean.to_csv(outputs / Path(out_csv).name, index=False)
    joblib.dump(models['kmeans'], outputs / "kmeans_final.joblib")
    joblib.dump(models['scaler'], outputs / "scaler.joblib")
    logger.info("Saved final CSV to outputs/%s (silhouette=%.4f)", Path(out_csv).name, float(sil))

    # show short cluster summary
    logger.info("Cluster distribution:")
    print(df_clean['cluster'].value_counts().sort_index().to_string())

    return outputs / Path(out_csv).name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv", help="Path to CSV")
    parser.add_argument("--out", type=str, default="outputs/netflix_with_clusters.csv", help="Output CSV path (under outputs/)")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for quick tests")
    parser.add_argument("--k", type=int, default=6, help="K for KMeans")
    parser.add_argument("--tfidf_max_features", type=int, default=3000, help="max features for TF-IDF")
    parser.add_argument("--svd_n_components", type=int, default=50, help="n components for TruncatedSVD")
    args = parser.parse_args()
    try:
        outp = main(args.csv, args.out, args.sample, args.k, args.tfidf_max_features, args.svd_n_components)
        logger.info("Pipeline finished. Output saved at: %s", outp)
    except Exception as e:
        logger.exception("Pipeline failed with exception: %s", e)
        raise
