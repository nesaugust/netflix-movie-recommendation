import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(
    page_title="Netflix Movie Recommendation App",
    page_icon="🎬",
    layout="wide"
)

# ================= CSS =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #000000 0%, #120000 45%, #000000 100%);
    color: white;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080808, #171717);
    border-right: 1px solid #2b2b2b;
}

h1 {
    font-size: 42px !important;
    line-height: 1.1;
    color: white;
}

h2, h3 {
    color: white;
}

.netflix-logo {
    color: #E50914;
    font-size: 38px;
    font-weight: 900;
    letter-spacing: 2px;
}

.hero {
    background: linear-gradient(90deg, rgba(0,0,0,0.95), rgba(120,0,0,0.35), rgba(0,0,0,0.95));
    padding: 35px;
    border-radius: 20px;
    border: 1px solid #2b2b2b;
    margin-bottom: 25px;
}

.metric-box {
    background: rgba(20,20,20,0.9);
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #333;
}

.movie-card {
    background: rgba(18,18,18,0.95);
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #333;
    margin-bottom: 18px;
}

.movie-card:hover {
    border: 1px solid #E50914;
}

.stButton>button {
    background-color: #E50914;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 25px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #b20710;
    color: white;
}

.footer {
    text-align: center;
    color: #aaa;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
df = pd.read_csv("clean_movies.csv")

df = df.dropna(subset=["Title", "Overview", "Genre"])
df["Title"] = df["Title"].astype(str)
df["Overview"] = df["Overview"].astype(str)
df["Genre"] = df["Genre"].astype(str)

if "Vote_Average" in df.columns:
    df["Vote_Average"] = pd.to_numeric(df["Vote_Average"], errors="coerce")

if "Popularity" in df.columns:
    df["Popularity"] = pd.to_numeric(df["Popularity"], errors="coerce")

# Remove duplicate titles to prevent index error
df = df.drop_duplicates(subset=["Title"]).reset_index(drop=True)

# ================= LANGUAGE =================
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
    "vi": "Vietnamese",
    "fa": "Persian",
    "hu": "Hungarian",
    "is": "Icelandic",
    "ms": "Malay",
    "sr": "Serbian",
    "tl": "Tagalog",
    "eu": "Basque",
    "et": "Estonian",
    "lv": "Latvian",
    "la": "Latin",
    "nb": "Norwegian Bokmål"
}

if "Original_Language" in df.columns:
    df["Language_Full"] = df["Original_Language"].map(language_map)
    df["Language_Full"] = df["Language_Full"].fillna("Other / Unknown")
else:
    df["Language_Full"] = "Unknown"

# ================= GENRE =================
all_genres = sorted(
    set(
        genre.strip()
        for genres in df["Genre"].dropna()
        for genre in genres.split(",")
    )
)

# ================= MODEL =================
df["Content"] = df["Overview"].str.lower() + " " + df["Genre"].str.lower()

tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
tfidf_matrix = tfidf.fit_transform(df["Content"])

indices = pd.Series(df.index, index=df["Title"]).drop_duplicates()

def recommend_movies(title, num_recommendations=5):
    if title not in indices:
        return pd.DataFrame()

    idx = int(indices[title])

    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

    # exclude selected movie itself
    sim_scores[idx] = -1

    top_indices = sim_scores.argsort()[-num_recommendations:][::-1]

    results = df.iloc[top_indices].copy()
    results["Similarity_Score"] = sim_scores[top_indices]

    return results

# ================= SIDEBAR =================
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
        options=all_genres
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
        ["All Languages"] + languages
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("This app recommends movies using NLP, TF-IDF, and content similarity.")

# ================= HEADER =================
st.markdown("""
<div class="hero">
    <div class="netflix-logo">NETFLIX</div>
    <h1>Movie Recommendation App</h1>
    <p>Find your next favorite movie based on storyline, genre, and content similarity.</p>
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

# ================= FILTER DATA =================
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

filtered_df = filtered_df[filtered_df["Vote_Average"].fillna(0) >= min_rating]

if selected_language != "All Languages":
    filtered_df = filtered_df[filtered_df["Language_Full"] == selected_language]

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Recommendation", "🔥 Trending", "⭐ Top Rated", "📊 Analytics"]
)

# ================= RECOMMENDATION =================
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
        st.warning("No movies found with selected filters.")
    else:
        movie_title = st.selectbox("Choose a movie:", movie_options)

        if st.button("Recommend"):
            recommendations = recommend_movies(movie_title, num_recommendations)

            if recommendations.empty:
                st.warning("No recommendations found.")
            else:
                st.markdown(f"## Recommendations for: **{movie_title}**")

                for _, row in recommendations.iterrows():
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)

                    col_img, col_text = st.columns([1, 4])

                    with col_img:
                        if "Poster_Url" in df.columns and pd.notna(row["Poster_Url"]):
                            st.image(row["Poster_Url"], width=160)

                    with col_text:
                        st.markdown(f"### {row['Title']}")
                        st.write(f"**Genre:** {row['Genre']}")
                        st.write(f"**Language:** {row['Language_Full']}")

                        if pd.notna(row["Vote_Average"]):
                            stars = "⭐" * int(float(row["Vote_Average"]) // 2)
                            st.write(f"**Rating:** {row['Vote_Average']} {stars}")

                        if "Popularity" in df.columns:
                            st.write(f"**Popularity:** {row['Popularity']}")

                        st.write(f"**Similarity:** {row['Similarity_Score'] * 100:.1f}%")
                        st.write(row["Overview"])

                    st.markdown("</div>", unsafe_allow_html=True)

# ================= TRENDING =================
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
                st.write(f"**Rating:** {row['Vote_Average']}")
                st.write(row["Overview"])

            st.markdown("</div>", unsafe_allow_html=True)

# ================= TOP RATED =================
with tab3:
    st.subheader("⭐ Top Rated Movies")

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

# ================= ANALYTICS =================
with tab4:
    st.subheader("📊 Movie Analytics")

    genre_df = df.copy()
    genre_df["Genre_List"] = genre_df["Genre"].str.split(",")
    genre_df = genre_df.explode("Genre_List")
    genre_df["Genre_List"] = genre_df["Genre_List"].str.strip()

    col1, col2 = st.columns(2)

    with col1:
        st.write("Top Genres")
        st.bar_chart(genre_df["Genre_List"].value_counts().head(10))

    with col2:
        st.write("Top Languages")
        st.bar_chart(df["Language_Full"].value_counts().head(10))

    st.write("Rating Distribution")
    st.bar_chart(df["Vote_Average"].round(1).value_counts().sort_index())

# ================= FOOTER =================
st.markdown("""
<div class="footer">
Built with Python, Pandas, Scikit-learn, NLP, TF-IDF, and Streamlit by Agnes Jeni
</div>
""", unsafe_allow_html=True)