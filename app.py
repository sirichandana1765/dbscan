import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent / "models"


@st.cache_data
def load_data() -> pd.DataFrame:
    path = DATA_DIR / "train.csv"
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_model():
    model_path = MODEL_DIR / "dbscan_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        return None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def add_background_style():
    st.set_page_config(
        page_title="Titanic DBSCAN Explorer",
        page_icon="🚢",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0b3c5d 0%, #1f7a8c 45%, #3bb78f 100%);
            color: #ffffff;
        }
        .stButton>button {
            background-color: #ff8c42;
            color: #0b0b0b;
            font-weight: 700;
        }
        .stMarkdown {
            color: #ffffff;
        }
        .css-1d391kg {
            background: rgba(255, 255, 255, 0.08);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            border-radius: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_clustered_data(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=["number"]).fillna(0)
    scaled = scaler.transform(numeric_df)
    labels = model.fit_predict(scaled)
    result = df.copy()
    result["cluster"] = labels
    return result


def display_metrics(df: pd.DataFrame):
    st.write("### Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)

    counts = df["Survived"].value_counts().rename({0: "Did Not Survive", 1: "Survived"})
    st.write("### Survival Summary")
    st.bar_chart(counts)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    st.write(f"**Numeric features used for clustering:** {', '.join(numeric_cols)}")


def display_cluster_analysis(clustered_df: pd.DataFrame):
    st.write("### DBSCAN Cluster Analysis")
    cluster_summary = (
        clustered_df.groupby("cluster")["PassengerId"].count().reset_index().rename(columns={"PassengerId": "count"})
    )
    st.dataframe(cluster_summary, use_container_width=True)
    st.bar_chart(cluster_summary.set_index("cluster")["count"])


def main():
    add_background_style()
    st.title("Titanic DBSCAN Explorer")
    st.markdown(
        "Explore Titanic passenger data with an unsupervised DBSCAN clustering workflow. "
        "Train the model using numeric features and inspect cluster structure from the Streamlit dashboard."
    )

    df = load_data()
    model, scaler = load_model()

    if model is None or scaler is None:
        st.warning("Model files are missing. Run `python train.py` to generate the DBSCAN model and scaler.")
    else:
        clustered_df = get_clustered_data(df, model, scaler)
        display_metrics(df)
        display_cluster_analysis(clustered_df)

    st.write("---")
    st.write(
        "This Streamlit app uses DBSCAN to reveal structure in Titanic numeric features. "
        "The background and layout are styled for a modern analytics experience."
    )


if __name__ == "__main__":
    main()
