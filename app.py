import streamlit as st
from knn import load_data, build_model, recommend_from_text
# importing the separate KNN model for Find Similar Perfumes
from knn_sp import (
    load_similar_data,
    build_similar_model,
    find_similar_perfumes
)


st.set_page_config(
    page_title="Perfume Finder",
    page_icon="🌸",
    layout="wide"
)


# Load the perfume dataset
@st.cache_data
def load_dataset():
    return load_similar_data(
        "final_perfume_data.csv"
    )


@st.cache_resource
def load_model(df):
    return build_similar_model(df)


try:
    df = load_dataset()
    vectorizer, tfidf_matrix, knn = load_model(df)

except FileNotFoundError:
    st.error(
        "final_perfume_data.csv was not found. "
        "Place it in the same folder as app.py."
    )
    st.stop()


st.title("Perfume Finder")

st.write(
    "Find perfumes based on scent notes, descriptions, "
    "and fragrance families."
)


st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Choose an option:",
    [
        "Find Similar Perfumes",
        "Search by Scent Preferences",
        "Explore Fragrance Families"
    ]
)


@st.cache_resource
def get_scent_knn():
    df = load_data()
    vectorizer, knn = build_model(df)
    return df, vectorizer, knn


# NEW: completed Find Similar Perfumes page
if page == "Find Similar Perfumes":

    st.header("Find Similar Perfumes")

    st.write(
        "Select a perfume to find similar fragrances."
    )

    perfume_names = sorted(
        df["Name"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_perfume = st.selectbox(
        "Choose a perfume:",
        perfume_names
    )

    number_of_results = st.slider(
        "Number of recommendations:",
        min_value=1,
        max_value=10,
        value=4
    )

    if st.button("Find Similar Perfumes"):

        recommendations = find_similar_perfumes(
            selected_perfume,
            df,
            tfidf_matrix,
            knn,
            number_of_results
        )

        st.subheader(
            f"Perfumes similar to {selected_perfume}"
        )

        for _, perfume in recommendations.iterrows():

            with st.container(border=True):

                st.subheader(perfume["Name"])

                st.write(
                    f"**Brand:** {perfume['Brand']}"
                )

                st.write(
                    f"**Similarity:** "
                    f"{perfume['similarity']:.1%}"
                )

                st.write(
                    f"**Notes:** {perfume['Notes']}"
                )


elif page == "Search by Scent Preferences":

    st.header("Search by Scent Preferences")

    st.write(
        "Enter scent notes such as sweet, floral, or vanilla."
    )
    scent_df, scent_vectorizer, scent_knn = get_scent_knn()

    query = st.text_input(
        "Scent notes:",
        placeholder="vanilla, jasmine, musk"
    )

    number_of_scent_results = st.slider(
        "Number of recommendations:",
        min_value=1,
        max_value=10,
        value=4,
        key="scent_slider"
    )

    if st.button("Search by Scent") and query.strip():

        try:
            results = recommend_from_text(
                scent_df,
                scent_vectorizer,
                scent_knn,
                query,
                number_of_scent_results
            )

            st.subheader(f"Perfumes matching: {query}")

            for _, perfume in results.iterrows():

                with st.container(border=True):

                    st.subheader(perfume["Name"])

                    st.write(
                        f"**Brand:** {perfume['Brand']}"
                    )

                    st.write(
                        f"**Similarity:** "
                        f"{perfume['similarity']:.1%}"
                    )

                    st.write(
                        f"**Notes:** {perfume['Notes']}"
                    )

        except ValueError as e:
            st.warning(str(e))

# K-Means 
elif page == "Explore Fragrance Families":

    st.header("Explore Fragrance Families")

    st.write(
        "The K-Means model will be connected "
        "to this page."
    )
