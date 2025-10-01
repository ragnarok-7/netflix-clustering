# Netflix Titles Clustering

## Quickstart
1. Create & activate venv:
   python -m venv venv
   .\venv\Scripts\Activate   # Windows
   pip install -r requirements.txt

2. Put dataset at:
   data/NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv

3. Run full pipeline:
   python scripts/run_pipeline.py --csv data/NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv --out outputs/netflix_with_clusters.csv --k 6

Or open notebooks in `notebooks/` and run in order: 01 -> 05.
