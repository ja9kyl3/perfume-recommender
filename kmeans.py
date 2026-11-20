#imports
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist


def load_data(path="final_perfume_data.csv"):
    df = pd.read_csv(path, encoding="latin1")

    df["Notes"] = (
        df["Notes"]
        .fillna("")
        .astype(str)
        .str.replace(
            r"\s*click\s+here\s+for ingredients?.*$",
            "",
            case=False,
            regex=True
        )
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    if "Description" in df.columns:
        df["combined_text"] = (
            df["Description"].fillna("").astype(str)
            + " "
            + df["Notes"]
        )
    else:
        df["combined_text"] = df["Notes"]

    df = df[df["combined_text"].str.strip() != ""]
    df = df.reset_index(drop=True)

    return df

def train_kmeans_model(df, n_clusters=5):
    """
    Train the TF-IDF and K-Means model
    """
    
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )
    
    tfidf_matrix = vectorizer.fit_transform(
        df["Notes"]
    )
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20
    )
    
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(tfidf_matrix)
    
    return df, vectorizer, tfidf_matrix, kmeans

def add_fragrance_family_names(df):
    """
    Assign fragrance family names to K-Means clusters.
    """
    
    cluster_family_names = {
        0: "Woody Leather Amber",
        1: "Spicy Woody Amber",
        2: "Citrus Floral Musk",
        3: "White Floral",
        4: "Sweet Woody Amber"
    }
    
    df = df.copy()
    
    df["fragrance_family"] = df["cluster"].map(
        cluster_family_names
    )
    
    return df

def get_cluster_keywords(
    vectorizer,
    kmeans,
    number_of_words=10
):
    feature_names = vectorizer.get_feature_names_out()
    cluster_keywords = {}

    for cluster_number in range(kmeans.n_clusters):
        center = kmeans.cluster_centers_[cluster_number]

        top_indices = center.argsort()[
            -number_of_words:
        ][::-1]

        cluster_keywords[cluster_number] = (
            feature_names[top_indices].tolist()
        )

    return cluster_keywords

def load_kmeans_model(
    path="final_perfume_data.csv",
    n_clusters=5
):
    df = load_data(path)
    df, vectorizer, tfidf_matrix, kmeans = (
        train_kmeans_model(
            df,
            n_clusters=n_clusters
        )
    )
    
    df = add_fragrance_family_names(df)
    
    print(
        pd.crosstab(
            df["cluster"],
            df["fragrance_family"]
        )
    )

    return df, vectorizer, tfidf_matrix, kmeans

