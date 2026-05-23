import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(
    page_title="Netflix Movie Recommendation App",
    page_icon="🎬",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.stApp {
    background-color: #141414;
    color: white;
}

h1, h2, h3 {
    color: #E50914;
}

.stButton>button {
    background-color: #E50914;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #b20710;
    color: white;
}

.movie-card {
    background-color: #1f1f1f;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 20px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# Load data
df = pd.read_csv("clean_movies.csv")

# Cleaning
df = df.dropna(subset=["Title", "Overview", "Genre"])
df["Overview"] = df["Overview"].astype(str)
df["Genre"] = df["Genre"].astype(str)
df["Content"] = df["Overview"].str.lower() + " " + df["Genre"].str.lower()
df = df.reset_index(drop=True)

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
tfidf_matrix = tfidf.fit_transform(df["Content"])

indices = pd.Series(df.index, index=df["Title"]).drop_duplicates()

def recommend_movies(title, num_recommendations=5):
    idx = indices[title]
    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-num_recommendations-1:-1][::-1]
    return df.iloc[top_indices]

# Sidebar
st.sidebar.title("⚙️ Settings")

num_recommendations = st.sidebar.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

genre_filter = st.sidebar.selectbox(
    "Filter by Genre",
    ["All"] + sorted(df["Genre"].dropna().unique().tolist())
)

# Header
st.title("🎬 Netflix Movie Recommendation App")
st.write("Find similar movies based on genre and storyline using NLP.")

# Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total Movies", len(df))
col2.metric("Total Genres", df["Genre"].nunique())
col3.metric("Languages", df["Original_Language"].nunique() if "Original_Language" in df.columns else "N/A")

st.divider()

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Recommendation", "🔥 Trending Movies", "📊 Analytics"])

with tab1:
    if genre_filter != "All":
        filtered_df = df[df["Genre"].str.contains(genre_filter, case=False, na=False)]
    else:
        filtered_df = df

    movie_title = st.selectbox(
        "Choose a movie:",
        filtered_df["Title"].sort_values()
    )

    if st.button("Recommend"):
        with st.spinner("Finding similar movies..."):
            recommendations = recommend_movies(movie_title, num_recommendations)

        st.subheader(f"Recommendations for: {movie_title}")

        for _, row in recommendations.iterrows():
            with st.container():
                col_img, col_text = st.columns([1, 4])

                with col_img:
                    if "Poster_Url" in df.columns and pd.notna(row["Poster_Url"]):
                        st.image(row["Poster_Url"], width=160)

                with col_text:
                    st.markdown(f"### {row['Title']}")
                    st.write(f"**Genre:** {row['Genre']}")

                    if "Vote_Average" in df.columns:
                        rating = row["Vote_Average"]
                        stars = "⭐" * int(float(rating) // 2)
                        st.write(f"**Rating:** {rating} {stars}")

                    if "Popularity" in df.columns:
                        st.write(f"**Popularity:** {row['Popularity']}")

                    st.write(row["Overview"])
                    st.caption("Recommended because it has similar storyline and genre.")

                st.divider()

with tab2:
    st.subheader("🔥 Top Trending Movies")

    if "Popularity" in df.columns:
        trending = df.sort_values("Popularity", ascending=False).head(10)

        for _, row in trending.iterrows():
            col_img, col_text = st.columns([1, 4])

            with col_img:
                if "Poster_Url" in df.columns and pd.notna(row["Poster_Url"]):
                    st.image(row["Poster_Url"], width=130)

            with col_text:
                st.markdown(f"### {row['Title']}")
                st.write(f"**Genre:** {row['Genre']}")
                st.write(f"**Popularity:** {row['Popularity']}")

                if "Vote_Average" in df.columns:
                    st.write(f"**Rating:** {row['Vote_Average']}")

            st.divider()
    else:
        st.warning("Popularity column not found.")

with tab3:
    st.subheader("📊 Movie Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Top Genres")
        genre_counts = df["Genre"].value_counts().head(10)
        st.bar_chart(genre_counts)

    with col2:
        if "Original_Language" in df.columns:
            st.write("Top Languages")
            language_counts = df["Original_Language"].value_counts().head(10)
            st.bar_chart(language_counts)

    if "Vote_Average" in df.columns:
        st.write("Rating Distribution")
        st.bar_chart(df["Vote_Average"].value_counts().sort_index())

st.markdown("""
<div class="footer">
Built with Python, NLP, TF-IDF, and Streamlit by Agnes Jeni
</div>
""", unsafe_allow_html=True)