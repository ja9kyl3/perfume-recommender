import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def load_similar_data(path="final_perfume_data.csv"):
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


def build_similar_model(df):
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        df["combined_text"]
    )

    knn = NearestNeighbors(
        n_neighbors=6,
        metric="cosine"
    )

    knn.fit(tfidf_matrix)

    return vectorizer, tfidf_matrix, knn


def find_similar_perfumes(
    perfume_name,
    df,
    tfidf_matrix,
    knn,
    number_of_results=4
):
    matches = df.index[
        df["Name"]
        .astype(str)
        .str.strip()
        .str.casefold()
        == perfume_name.strip().casefold()
    ].tolist()

    if not matches:
        return pd.DataFrame()

    selected_index = matches[0]

    distances, indices = knn.kneighbors(
        tfidf_matrix[selected_index],
        n_neighbors=min(
            number_of_results + 1,
            len(df)
        )
    )

    results = df.iloc[indices[0]].copy()

    results["similarity"] = 1 - distances[0]

    # Remove the perfume the user selected
    results = results[
        results.index != selected_index
    ]

    return results.head(number_of_results)