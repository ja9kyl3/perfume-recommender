import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DATA_PATH = "final_perfume_data.csv"

def load_data(path=DATA_PATH):
  df = pd.read_csv(path, encoding="latin1")
  df["Notes"] = (
    df["Notes"]
    .astype("string")
    .replace(
        r"^\s*(nan|none|null)\s*$",
        pd.NA,
        regex=True
      )
  )
  df = df.dropna(subset=["Notes"]).copy()
  df["Notes"] = (
    df["Notes"]
    .str.replace(
        r"\s*click\s+here\s+for\s+ingredients?.*$",
        "",
        case=False,
        regex=True
    )
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip(" ,.;:-")
  )
  df = df[df["Notes"] != ""].reset_index(drop=True)
  return df
  

def build_model(df, k=4):
  vectorizer = TfidfVectorizer(
      stop_words="english"
  )
  tfidf_matrix = vectorizer.fit_transform(df["Notes"])
  knn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(tfidf_matrix)
  return vectorizer, knn

def recommend_from_text(df, vectorizer, knn, query, k=4):
    query_vector = vectorizer.transform([query.lower()])
    if query_vector.nnz == 0:
      raise ValueError("None of those scents appear in dataset. Try again")
    distances, indices = knn.kneighbors(
        query_vector,
        n_neighbors=min(k, len(df))
    )
    result = df.iloc[indices[0]].copy()
    result['similarity'] = 1 - distances[0]
    return result[
        ['Name', 'Brand', 'Notes', 'similarity', 'Image URL']
    ]
