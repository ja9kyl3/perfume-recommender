
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Perfume Details",
    page_icon="🌸",
    layout="wide"
)

# Page styling
st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}
h1 {
    font-size: 2.5rem;
}
div[data-testid="stImage"] img {
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)


# Load perfume information
@st.cache_data
def load_perfumes():
    return pd.read_csv(
        "final_perfume_data.csv",
        encoding="latin1"
    ).fillna("")


df = load_perfumes()

# Back button
if st.button("← Back to Fragrance Families"):
    st.switch_page("app.py")

st.divider()

# Find the selected perfume
perfume_name = st.session_state.get(
    "selected_family_perfume"
)

if not perfume_name:
    st.warning("Please select a perfume first.")

else:
    matches = df[df["Name"] == perfume_name]

    if matches.empty:
        st.error("Perfume not found.")

    else:
        perfume = matches.iloc[0]

        # Two-column layout
        image_col, info_col = st.columns([1, 1.5], gap="large")

        with image_col:
            image_url = str(perfume.get("Image URL", ""))

            if image_url.startswith(("http://", "https://")):
                st.image(image_url, width="stretch")
            else:
                st.info("No perfume image available.")

        with info_col:
            st.caption(str(perfume["Brand"]).upper())

            st.title(perfume["Name"])

            st.divider()

            st.subheader("🌿 Fragrance Notes")

            # Display notes as small labels
            notes = str(perfume.get("Notes", "")).split(",")

            for note in notes:
                note = note.strip()

                if note:
                    st.badge(note)

        # Description section
        st.divider()
        st.subheader("About This Fragrance")

        description = str(
            perfume.get("Description", "")
        ).strip()

        if description:
            st.write(description)
        else:
            st.write("No description available.")
