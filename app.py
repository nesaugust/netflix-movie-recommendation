import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

st.set_page_config(page_title="Netflix Movie Recommendation App", layout="wide")

st.title("Netflix Movie Recommendation App")

# Load data
df = pd.read_csv("Netflix/clean_movies.csv")

# Basic cleaning
df = df.dropna(subset=["Title", "Overview", "Genre"])
df["Overview"] = df["Overview"].astype(str)
df["Genre"] = df["Genre"].astype(str)

# Combine text for recommendation
df["Content"] = df["Overview"].str.lower() + " " + df["Genre"].str.lower()

# Reset index
df = df.reset_index(drop=True)

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
tfidf_matrix = tfidf.fit_transform(df["Content"])

# Movie index
indices = pd.Series(df.index, index=df["Title"]).drop_duplicates()

def recommend_movies(title, num_recommendations=5):
    idx = indices[title]

    # Calculate similarity only for selected movie
    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Get top similar movies, excluding itself
    top_indices = sim_scores.argsort()[-num_recommendations-1:-1][::-1]

    return df.iloc[top_indices]

# Sidebar
st.sidebar.header("Settings")
num_recommendations = st.sidebar.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

# Movie selector
movie_title = st.selectbox(
    "Choose a movie:",
    df["Title"].sort_values()
)

if st.button("Recommend"):
    recommendations = recommend_movies(movie_title, num_recommendations)

    st.subheader(f"Recommendations for: {movie_title}")

    for _, row in recommendations.iterrows():
        col1, col2 = st.columns([1, 4])

        with col1:
            if "Poster_Url" in df.columns and pd.notna(row["Poster_Url"]):
                st.image(row["Poster_Url"], width=150)

        with col2:
            st.markdown(f"### {row['Title']}")
            st.write(f"**Genre:** {row['Genre']}")

            if "Vote_Average" in df.columns:
                st.write(f"**Rating:** {row['Vote_Average']}")

            if "Popularity" in df.columns:
                st.write(f"**Popularity:** {row['Popularity']}")

            st.write(row["Overview"])

        st.divider()



#streamlit run Netflix/app.py : to deploy the app