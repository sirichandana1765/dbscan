import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent / "models"


def load_data() -> pd.DataFrame:
    train_path = DATA_DIR / "train.csv"
    df = pd.read_csv(train_path)
    return df


def train_dbscan(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=["number"]).fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)

    model = DBSCAN(eps=1.0, min_samples=5, metric="euclidean")
    model.fit(scaled)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "dbscan_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    labels = model.labels_
    unique_clusters = len(set(labels))
    print(f"Saved DBSCAN model and scaler to {MODEL_DIR}")
    print(f"Detected {unique_clusters} clusters (including noise).")

    return model, scaler


def main():
    df = load_data()
    train_dbscan(df)


if __name__ == "__main__":
    main()
