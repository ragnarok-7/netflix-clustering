# scripts/build_features.py
import argparse
from scripts.utils import safe_load_csv, ensure_outputs_dir, logger
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned", default="outputs/cleaned_netflix.csv")
    parser.add_argument("--outx", default="outputs/X_combined.npy")
    args = parser.parse_args()

    df = safe_load_csv(args.cleaned)
    descriptions = df['description'].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_df=0.8, min_df=5, max_features=3000, ngram_range=(1,2))
    X_text = tfidf.fit_transform(descriptions)
    svd_n = 50 if X_text.shape[0] > 200 else min(20, X_text.shape[1])
    svd = TruncatedSVD(n_components=svd_n, random_state=42)
    X_text_red = svd.fit_transform(X_text)

    mlb = MultiLabelBinarizer(sparse_output=False)
    X_genres = mlb.fit_transform(df['genres_list'])

    duration = df.get('duration_num', np.nan).fillna(-1).values.reshape(-1,1)
    release_year = df.get('release_year', np.nan).fillna(-1).values.reshape(-1,1)

    X = np.hstack([X_text_red, X_genres, duration, release_year])
    np.save(args.outx, X)
    joblib.dump(tfidf, "outputs/tfidf.joblib")
    joblib.dump(svd, "outputs/svd.joblib")
    joblib.dump(mlb, "outputs/mlb.joblib")
    logger.info("Saved X to %s and vectorizers to outputs/", args.outx)
