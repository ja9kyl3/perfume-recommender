import streamlit as st
from knn import load_data, build_model, recommend_from_text
# importing the separate KNN model for Find Similar Perfumes
from knn_sp import (
    load_similar_data,
    build_similar_model,
    find_similar_perfumes
)

# import kmeans
from kmeans import (
    load_kmeans_model,
    get_cluster_keywords
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

@st.cache_resource
def get_model():
    return load_kmeans_model(
        path="final_perfume_data.csv",
        n_clusters=5
    )

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

                left, right = st.columns([3, 1])

                with left:
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

                with right:
                    st.image(perfume["Image URL"], width=150)


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

                    left, right = st.columns([3, 1])

                    with left:
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

                    with right:
                        st.image(perfume["Image URL"], width=150)

        except ValueError as e:
            st.warning(str(e))

# K-Means 
elif page == "Explore Fragrance Families":

    st.header("Explore Fragrance Families")

    st.write(
        "Select a scent group to explore perfumes with similar "
        "fragrance patterns."
    )
    
    df, vectorizer, tfidf_matrix, kmeans = get_model()
    
    cluster_keywords = get_cluster_keywords(
        vectorizer,
        kmeans,
        number_of_words=10
    )
        
    fragrance_families = sorted(
        df["fragrance_family"]
        .dropna()
        .unique()
        .tolist()
    )
    
    for cluster_number, words in cluster_keywords.items():
        family_name = (
            df[df["cluster"] == cluster_number]
            ["fragrance_family"]
            .iloc[0]
        )

        print(
            f"Cluster {cluster_number} "
            f"({family_name}):"
        )
        print(", ".join(words))
        print()
    
    family_options = ["View All Groups"] + fragrance_families
    
    selected_family = st.selectbox(
        "Choose a fragrance family",
        family_options
    )
    
    if selected_family == "View All Groups":
        st.subheader("All Fragrance Groups")
        
        group_summary = (
            df.groupby("fragrance_family")
            .size()
            .reset_index(name="number_of_perfumes")
            .sort_values(
                by="number_of_perfumes",
                ascending=False
            )
        )
        
        st.dataframe(
            group_summary,
            hide_index=True,
            width="stretch"
        )
        
        for family in fragrance_families:
            st.write(f"### {family}")
            
            family_perfumes = df[
                df["fragrance_family"] == family
            ][
                ["Name", "Brand", "Notes"]
            ]
            
            st.dataframe(
                family_perfumes.head(10),
                hide_index=True,
                width="stretch"
            )
    else:
        st.subheader(f"{selected_family} Perfumes")
        
        selected_perfumes = df[
            df["fragrance_family"] == selected_family
        ][
            ["Name", "Brand", "Notes", "cluster"]
        ]
        
        st.write(
            f"Number of perfumes: "
            f"{len(selected_perfumes)}"
        )
        
        st.dataframe(
            selected_perfumes,
            hide_index=True,
            width="stretch"
        )
