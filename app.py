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
# CSS DESIGN
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 70% 10%, rgba(229,9,20,0.22), transparent 25%),
        linear-gradient(135deg, #050505 0%, #000000 45%, #090000 100%);
    color: white;
}

.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101010, #181818);
    border-right: 1px solid #2b2b2b;
}

.topbar {
    border-top: 2px solid #E50914;
    border-bottom: 1px solid #2b2b2b;
    padding: 24px 24px 18px 24px;
    margin: 0 0 30px 0;
    background: #050505;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.netflix-brand {
    color: #E50914;
    font-size: 34px;
    font-weight: 900;
    letter-spacing: 2px;
}

.top-title {
    color: white;
    font-size: 18px;
    margin-left: 12px;
    font-weight: 600;
}

.hero {
    background:
        linear-gradient(90deg, rgba(0,0,0,0.95), rgba(80,0,0,0.65), rgba(0,0,0,0.85)),
        url("https://images.unsplash.com/photo-1524985069026-dd778a71c7b4");
    background-size: cover;
    background-position: center;
    padding: 55px 40px;
    border-radius: 0px;
    border: 1px solid #222;
    margin-bottom: 24px;
}

.hero-small {
    color: #bfbfbf;
    letter-spacing: 3px;
    font-size: 14px;
    font-weight: 600;
}

.hero-n {
    color: #E50914;
    font-size: 38px;
    font-weight: 900;
    margin-right: 8px;
}

.hero h1 {
    font-size: 46px !important;
    line-height: 1.08;
    font-weight: 800;
    color: white;
    margin-top: 18px;
    margin-bottom: 18px;
}

.hero p {
    color: #cccccc;
    font-size: 18px;
    max-width: 520px;
}

.metric-card {
    background: rgba(20,20,20,0.92);
    border: 1px solid #333;
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 22px;
}

.metric-label {
    color: #bfbfbf;
    font-size: 14px;
}

.metric-value {
    color: white;
    font-size: 34px;
    font-weight: 700;
}

.recommend-box {
    background: rgba(0,0,0,0.72);
    border: 1px solid #262626;
    border-radius: 10px;
    padding: 24px;
    margin-top: 20px;
}

.info-box {
    border-left: 3px solid #E50914;
    padding: 12px 18px;
    color: #cccccc;
    margin-top: 20px;
}

.movie-card {
    background: rgba(18,18,18,0.96);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #333;
    margin-bottom: 18px;
    transition: 0.25s;
}

.movie-card:hover {
    border: 1px solid #E50914;
    transform: translateY(-2px);
}

.genre-card {
    background: rgba(25,25,25,0.95);
    border: 1px solid #333;
    padding: 18px;
    border-radius: 10px;
    text-align: center;
}

.genre-count {
    color: #E50914;
    font-size: 14px;
}

.stButton>button {
    background: linear-gradient(90deg, #E50914, #b20710);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 28px;
    font-weight: 700;
}

.stButton>button:hover {
    background: #E50914;
    color: white;
    transform: scale(1.02);
}

div[data-baseweb="tab-list"] {
    background: rgba(15,15,15,0.95);
    border: 1px solid #2a2a2a;
    border-radius: 10px 10px 0 0;
    padding: 0px;
}

button[data-baseweb="tab"] {
    padding: 18px 28px;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #999;
    border-top: 1px solid #E50914;
    padding: 24px;
    margin-top: 50px;
}

.sidebar-card {
    border: 1px solid #333;
    border-radius: 10px;
    padding: 18px;
    margin-top: 20px;
    background: rgba(20,20,20,0.75);
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

# Fix duplicate title error
df = df.drop_duplicates(subset=["Title"]).reset_index(drop=True)

# =========================
# LANGUAGE MAPPING
# =========================
language_map = {
    "en": "English", "ko": "Korean", "ja": "Japanese", "fr": "French",
    "es": "Spanish", "de": "German", "it": "Italian", "zh": "Chinese",
    "cn": "Chinese", "hi": "Hindi", "id": "Indonesian", "ru": "Russian",
    "th": "Thai", "pt": "Portuguese", "tr": "Turkish", "ar": "Arabic",
    "da": "Danish", "sv": "Swedish", "nl": "Dutch", "pl": "Polish",
    "no": "Norwegian", "fi": "Finnish", "cs": "Czech", "bn": "Bengali",
    "ca": "Catalan", "te": "Telugu", "ta": "Tamil", "ml": "Malayalam",
    "mr": "Marathi", "el": "Greek", "he": "Hebrew", "ro": "Romanian",
    "uk": "Ukrainian", "vi": "Vietnamese", "fa": "Persian",
    "hu": "Hungarian", "is": "Icelandic", "ms": "Malay",
    "sr": "Serbian", "tl": "Tagalog", "eu": "Basque",
    "et": "Estonian", "lv": "Latvian", "la": "Latin",
    "nb": "Norwegian Bokmål"
}

if "Original_Language" in df.columns:
    df["Language_Full"] = df["Original_Language"].map(language_map).fillna("Other / Unknown")
else:
    df["Language_Full"] = "Unknown"

# =========================
# GENRES
# =========================
all_genres = sorted(
    set(
        genre.strip()
        for genres in df["Genre"].dropna()
        for genre in genres.split(",")
    )
)

# =========================
# MODEL
# =========================
df["Content"] = df["Overview"].str.lower() + " " + df["Genre"].str.lower()

tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
tfidf_matrix = tfidf.fit_transform(df["Content"])

indices = pd.Series(df.index, index=df["Title"]).drop_duplicates()

def recommend_movies(title, num_recommendations=5):
    if title not in indices:
        return pd.DataFrame()

    idx = int(indices[title])
    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = -1

    top_indices = sim_scores.argsort()[-num_recommendations:][::-1]

    results = df.iloc[top_indices].copy()
    results["Similarity_Score"] = sim_scores[top_indices]
    return results

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## ⚙️ SETTINGS")

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
        ["All Languages"] + languages
    )

    st.markdown("""
    <div class="sidebar-card">
        <h4 style="color:#E50914;">ⓘ ABOUT</h4>
        <p style="color:#ccc;">
        This app uses NLP and TF-IDF Vectorization to recommend movies similar to your favorite ones based on storyline, genre, and content similarity.
        </p>
        <hr>
        <h4 style="color:#E50914;">BUILT WITH</h4>
        <p>🐍 Python</p>
        <p>🐼 Pandas</p>
        <p>🤖 Scikit-learn</p>
        <p>🎬 Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# TOP BAR
# =========================
st.markdown("""
<div class="topbar">
    <div>
        <span class="netflix-brand">NETFLIX</span>
        <span class="top-title">| &nbsp; Movie Recommendation App</span>
    </div>
    <div style="color:#bbb;">🎬 Find your next favorite movie</div>
</div>
""", unsafe_allow_html=True)

# =========================
# HERO
# =========================
st.markdown("""
<div class="hero">
    <div>
        <span class="hero-n">N</span>
        <span class="hero-small">MOVIE RECOMMENDATION APP</span>
    </div>
    <h1>Find Your Next<br>Favorite Movie</h1>
    <p>Discover similar movies based on storyline, genre, and content similarity using NLP.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# METRICS
# =========================
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🎞️ Total Movies</div>
        <div class="metric-value">{len(df):,}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🎭 Total Genres</div>
        <div class="metric-value">{len(all_genres)}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🌐 Languages</div>
        <div class="metric-value">{df["Language_Full"].nunique()}</div>
    </div>
    """, unsafe_allow_html=True)

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

filtered_df = filtered_df[filtered_df["Vote_Average"].fillna(0) >= min_rating]

if selected_language != "All Languages":
    filtered_df = filtered_df[filtered_df["Language_Full"] == selected_language]

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Recommendation", "🔥 Trending", "⭐ Top Rated", "📊 Analytics"]
)

# =========================
# RECOMMENDATION
# =========================
with tab1:
    st.markdown('<div class="recommend-box">', unsafe_allow_html=True)
    st.subheader("🎯 Movie Recommendation")

    search_query = st.text_input(
        "Search for a movie",
        placeholder="Type a movie title..."
    )

    if search_query:
        movie_options = filtered_df[
            filtered_df["Title"].str.contains(search_query, case=False, na=False)
        ]["Title"].sort_values()
    else:
        movie_options = filtered_df["Title"].sort_values()

    if len(movie_options) == 0:
        st.warning("No movies found with selected filters.")
    else:
        c1, c2 = st.columns([4, 1])

        with c1:
            movie_title = st.selectbox(
                "Choose a movie",
                movie_options,
                index=None,
                placeholder="Select a movie..."
            )

        with c2:
            st.write("")
            st.write("")
            recommend_button = st.button("Recommend", use_container_width=True)

        st.markdown("""
        <div class="info-box">
            <b style="color:#E50914;">ⓘ How it works?</b><br>
            We analyze the plot overview and genres using TF-IDF Vectorization and cosine similarity to find the most similar movies for you.
        </div>
        """, unsafe_allow_html=True)

        if recommend_button:
            if not movie_title:
                st.warning("Please choose a movie first.")
            else:
                recommendations = recommend_movies(movie_title, num_recommendations)

                st.markdown(f"### Recommendations for: <span style='color:#E50914'>{movie_title}</span>", unsafe_allow_html=True)

                for _, row in recommendations.iterrows():
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)

                    col_img, col_text = st.columns([1, 4])

                    with col_img:
                        if "Poster_Url" in df.columns and pd.notna(row["Poster_Url"]):
                            st.image(row["Poster_Url"], width=145)

                    with col_text:
                        st.markdown(f"### {row['Title']}")
                        st.write(f"**Genre:** {row['Genre']}")
                        st.write(f"**Language:** {row['Language_Full']}")
                        st.write(f"**Rating:** {row['Vote_Average']} ⭐")
                        st.write(f"**Similarity:** {row['Similarity_Score'] * 100:.1f}% Match")
                        st.write(row["Overview"])

                    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TRENDING
# =========================
with tab2:
    st.subheader("🔥 Trending Movies")

    trending = filtered_df.sort_values("Popularity", ascending=False).head(10)

    for _, row in trending.iterrows():
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        col_img, col_text = st.columns([1, 4])

        with col_img:
            if "Poster_Url" in df.columns and pd.notna(row["Poster_Url"]):
                st.image(row["Poster_Url"], width=130)

        with col_text:
            st.markdown(f"### {row['Title']}")
            st.write(f"**Genre:** {row['Genre']}")
            st.write(f"**Language:** {row['Language_Full']}")
            st.write(f"**Popularity:** {row['Popularity']}")
            st.write(f"**Rating:** {row['Vote_Average']} ⭐")
            st.write(row["Overview"])

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TOP RATED
# =========================
with tab3:
    st.subheader("⭐ Top Rated Movies")

    top_rated = filtered_df.sort_values("Vote_Average", ascending=False).head(10)

    for _, row in top_rated.iterrows():
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        col_img, col_text = st.columns([1, 4])

        with col_img:
            if "Poster_Url" in df.columns and pd.notna(row["Poster_Url"]):
                st.image(row["Poster_Url"], width=130)

        with col_text:
            st.markdown(f"### {row['Title']}")
            st.write(f"**Genre:** {row['Genre']}")
            st.write(f"**Language:** {row['Language_Full']}")
            st.write(f"**Rating:** {row['Vote_Average']} ⭐")
            st.write(row["Overview"])

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# ANALYTICS
# =========================
with tab4:
    st.subheader("📊 Movie Analytics")

    genre_df = df.copy()
    genre_df["Genre_List"] = genre_df["Genre"].str.split(",")
    genre_df = genre_df.explode("Genre_List")
    genre_df["Genre_List"] = genre_df["Genre_List"].str.strip()

    c1, c2 = st.columns(2)

    with c1:
        st.write("Top Individual Genres")
        st.bar_chart(genre_df["Genre_List"].value_counts().head(10))

    with c2:
        st.write("Top Languages")
        st.bar_chart(df["Language_Full"].value_counts().head(10))

    st.write("Rating Distribution")
    st.bar_chart(df["Vote_Average"].round(1).value_counts().sort_index())

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
© 2025 Netflix Movie Recommendation App • Built with <span style="color:#E50914;">Streamlit</span> by Agnes Jeni
</div>
""", unsafe_allow_html=True)