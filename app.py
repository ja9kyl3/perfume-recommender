import streamlit as st

st.set_page_config(
    page_title="Perfume Finder",
    page_icon="🌸",
    layout="wide"
)

st.title("Perfume Finder")
st.write("Find perfumes based on scent notes, descriptions, and fragrance families.")

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Choose an option:",
    [
        "Find Similar Perfumes",
        "Search by Scent Preferences",
        "Explore Fragrance Families"
    ]
)

if page == "Find Similar Perfumes":
    st.header("Find Similar Perfumes")
    st.write("Select a perfume to find similar fragrances.")

elif page == "Search by Scent Preferences":
    st.header("Search by Scent Preferences")
    st.write("Enter scent notes such as sweet, floral, or vanilla.")

elif page == "Explore Fragrance Families":
    st.header("Explore Fragrance Families")
    st.write("Explore fragrance groups created using K-Means clustering.")