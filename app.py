import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(
    page_title="Netflix Movie Recommendation App",
    page_icon="🎬",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.stApp {
    background:
        linear-gradient(rgba(0,0,0,0.86), rgba(0,0,0,0.92)),
        url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111111, #1f1f1f);
    border-right: 1px solid #333333;
}

h1 {
    color: #ffffff;
    font-size: 48px;
    font-weight: 800;
}

h2, h3 {
    color: #ffffff;
}

.stButton>button {
    background-color: #E50914;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 26px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    background-color: #b20710;
    color: white;
    transform: scale(1.03);
}

.movie-card {
    background: rgba(31, 31, 31, 0.88);
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 24px;
    border: 1px solid #333333;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.45);
}

.movie-card:hover {
    border: 1px solid #E50914;
}

.hero-box {
    background: rgba(0,0,0,0.58);
    padding: 35px;
    border-radius: 22px;
    margin-bottom: 25px;
    border: 1px solid #333333;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.55);
}

.footer {
    text-align: center;
    color: #aaaaaa;
    margin-top: 50px;
    font-size: 14px;
}

.small-text {
    color: #bbbbbb;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("clean_movies.csv")

df = df.dropna(subset=["Title", "Overview", "Genre"])
df["Title"] = df["Title"].astype(str)
df["Overview"] = df["Overview"].astype(str)
df["Genre"] = df["Genre"].astype(str)

if "Vote_Average" in df.columns:
    df["Vote_Average"] = pd.to_numeric(df["Vote_Average"], errors="coerce")

if "Popularity" in df.columns:
    df["Popularity"] = pd.to_numeric(df["Popularity"], errors="coerce")

# =========================
# LANGUAGE FULL NAME
# =========================
language_map = {
    "en": "English",
    "ko": "Korean",
    "ja": "Japanese",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "zh": "Chinese",
    "cn": "Chinese",
    "hi": "Hindi",
    "id": "Indonesian",
    "ru": "Russian",
    "th": "Thai",
    "pt": "Portuguese",
    "tr": "Turkish",
    "ar": "Arabic",
    "da": "Danish",
    "sv": "Swedish",
    "nl": "Dutch",
    "pl": "Polish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "bn": "Bengali",
    "ca": "Catalan",
    "te": "Telugu",
    "ta": "Tamil",
    "ml": "Malayalam",
    "mr": "Marathi",
    "el": "Greek",
    "he": "Hebrew",
    "ro": "Romanian",
    "uk": "Ukrainian",
    "vi": "Vietnamese"
}

df["Language_Full"] = df["Original_Language"].map(language_map).fillna(df["Original_Language"])

df["Content"] = df["Overview"].str.lower() + " " + df["Genre"].str.lower()
df = df.reset_index(drop=True)

# =========================
# GENRE LIST
# =========================
all_genres = sorted(
    set(
        genre.strip()
        for genres in df["Genre"].dropna()
        for genre in genres.split(",")
    )
)

# =========================
# TF-IDF MODEL
# =========================
tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
tfidf_matrix = tfidf.fit_transform(df["Content"])
indices = pd.Series(df.index, index=df["Title"]).drop_duplicates()

def recommend_movies(title, num_recommendations=5):
    idx = indices[title]
    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-num_recommendations-1:-1][::-1]
    scores = sim_scores[top_indices]

    results = df.iloc[top_indices].copy()
    results["Similarity_Score"] = scores
    return results

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    num_recommendations = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

    selected_genres = st.multiselect(
        "Filter by Genre",
        options=all_genres,
        default=[]
    )

    min_rating = st.slider(
        "Minimum Rating",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.5
    )

    languages = sorted(df["Language_Full"].dropna().unique().tolist())

    selected_language = st.selectbox(
        "Filter by Language",
        ["All"] + languages
    )

    st.markdown("---")
    st.caption("Built using NLP, TF-IDF, and Streamlit")

# =========================
# HERO
# =========================
st.markdown("""
<div class="hero-box">
    <h1>🎬 Netflix Movie Recommendation App</h1>
    <p class="small-text">
        Discover similar movies based on storyline, genre, and content similarity using NLP.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Movies", len(df))

with col2:
    st.metric("Total Genres", len(all_genres))

with col3:
    st.metric("Languages", df["Language_Full"].nunique())

st.divider()

# =========================
# FILTER DATA
# =========================
filtered_df = df.copy()

if selected_genres:
    filtered_df = filtered_df[
        filtered_df["Genre"].apply(
            lambda x: any(
                genre in [g.strip() for g in x.split(",")]
                for genre in selected_genres
            )
        )
    ]

if "Vote_Average" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Vote_Average"] >= min_rating]

if selected_language != "All":
    filtered_df = filtered_df[filtered_df["Language_Full"] == selected_language]

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Recommendation", "🔥 Trending", "⭐ Top Rated", "📊 Analytics"]
)

# =========================
# RECOMMENDATION TAB
# =========================
with tab1:
    st.subheader("🎯 Movie Recommendation")

    search_query = st.text_input("Search movie title")

    if search_query:
        movie_options = filtered_df[
            filtered_df["Title"].str.contains(search_query, case=False, na=False)
        ]["Title"].sort_values()
    else:
        movie_options = filtered_df["Title"].sort_values()

    if len(movie_options) == 0:
        st.warning("No movies found with the selected filters.")
    else:
        movie_title = st.selectbox("Choose a movie:", movie_options)

        if st.button("Recommend"):
            with st.spinner("Finding similar movies..."):
                recommendations = recommend_movies(movie_title, num_recommendations)

            st.markdown(f"## Recommendations for: **{movie_title}**")

            for _, row in recommendations.iterrows():
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)

                col_img, col_text = st.columns([1, 4])

                with col_img:
                    if "Poster_Url" in df.columns and pd.notna(row["Poster_Url"]):
                        st.image(row["Poster_Url"], width=170)
                    else:
                        st.write("No poster available")

                with col_text:
                    st.markdown(f"### {row['Title']}")
                    st.write(f"**Genre:** {row['Genre']}")
                    st.write(f"**Language:** {row['Language_Full']}")

                    if "Vote_Average" in df.columns:
                        rating = row["Vote_Average"]
                        stars = "⭐" * int(float(rating) // 2) if pd.notna(rating) else ""
                        st.write(f"**Rating:** {rating} {stars}")

                    if "Popularity" in df.columns:
                        st.write(f"**Popularity:** {row['Popularity']}")

                    st.write(f"**Similarity Score:** {row['Similarity_Score']:.2f}")
                    st.write(row["Overview"])
                    st.caption("Recommended because it has similar storyline and genre.")

                st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TRENDING TAB
# =========================
with tab2:
    st.subheader("🔥 Trending Movies")

    if "Popularity" in df.columns:
        trending = filtered_df.sort_values("Popularity", ascending=False).head(10)

        for _, row in trending.iterrows():
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)

            col_img, col_text = st.columns([1, 4])

            with col_img:
                if "Poster_Url" in df.columns and pd.notna(row["Poster_Url"]):
                    st.image(row["Poster_Url"], width=140)

            with col_text:
                st.markdown(f"### {row['Title']}")
                st.write(f"**Genre:** {row['Genre']}")
                st.write(f"**Language:** {row['Language_Full']}")
                st.write(f"**Popularity:** {row['Popularity']}")

                if "Vote_Average" in df.columns:
                    st.write(f"**Rating:** {row['Vote_Average']}")

                st.write(row["Overview"])

            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Popularity column not found.")

# =========================
# TOP RATED TAB
# =========================
with tab3:
    st.subheader("⭐ Top Rated Movies")

    if "Vote_Average" in df.columns:
        top_rated = filtered_df.sort_values("Vote_Average", ascending=False).head(10)

        for _, row in top_rated.iterrows():
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)

            col_img, col_text = st.columns([1, 4])

            with col_img:
                if "Poster_Url" in df.columns and pd.notna(row["Poster_Url"]):
                    st.image(row["Poster_Url"], width=140)

            with col_text:
                st.markdown(f"### {row['Title']}")
                st.write(f"**Genre:** {row['Genre']}")
                st.write(f"**Language:** {row['Language_Full']}")
                st.write(f"**Rating:** {row['Vote_Average']} ⭐")
                st.write(row["Overview"])

            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Vote_Average column not found.")

# =========================
# ANALYTICS TAB
# =========================
with tab4:
    st.subheader("📊 Movie Analytics")

    genre_exploded = df.copy()
    genre_exploded["Genre_List"] = genre_exploded["Genre"].str.split(",")
    genre_exploded = genre_exploded.explode("Genre_List")
    genre_exploded["Genre_List"] = genre_exploded["Genre_List"].str.strip()

    col1, col2 = st.columns(2)

    with col1:
        st.write("Top Individual Genres")
        genre_counts = genre_exploded["Genre_List"].value_counts().head(10)
        st.bar_chart(genre_counts)

    with col2:
        st.write("Top Languages")
        language_counts = df["Language_Full"].value_counts().head(10)
        st.bar_chart(language_counts)

    if "Vote_Average" in df.columns:
        st.write("Rating Distribution")
        rating_counts = df["Vote_Average"].round(1).value_counts().sort_index()
        st.bar_chart(rating_counts)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
Built with Python, Pandas, Scikit-learn, NLP, TF-IDF, and Streamlit by Agnes Jeni
</div>
""", unsafe_allow_html=True)