# Netflix Movie Recommendation App

A modern Netflix-inspired movie recommendation system built using Python, Streamlit, Scikit-learn, and Natural Language Processing (NLP).

This application recommends similar movies based on storyline, genre, and content similarity using TF-IDF vectorization and cosine similarity.

---

## Live Demo

https://netflix-movie-recommendation-i3cy5cdmd6ddufygcmkql4.streamlit.app/

---

## GitHub Repository

https://github.com/nesaugust/netflix-movie-recommendation

---

## Features

- Netflix-inspired modern UI
- Movie recommendation system using NLP
- Search movies by title
- Filter movies by:
  - Genre
  - Rating
  - Language
- Trending movies section
- Top rated movies section
- Interactive analytics dashboard
- Movie poster display
- Responsive dark theme interface
- Similarity score recommendation
- Content-based recommendation engine

---

## How It Works

The recommendation engine combines movie overview and genre information into a single text feature.

The text data is transformed into numerical vectors using TF-IDF Vectorization. Cosine similarity is then calculated between movies to identify and recommend the most similar titles.

---

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Natural Language Processing (NLP)

---

## Dataset

The dataset includes:

- Movie Title
- Overview
- Genre
- Popularity
- Vote Average
- Original Language
- Poster URL

---

## Project Structure

```text
Netflix-Movie-Recommendation/
│
├── app.py
├── clean_movies.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Run Locally

Clone the repository:

```bash
git clone https://github.com/nesaugust/netflix-movie-recommendation.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## Recommendation System

This project uses a content-based filtering recommendation system powered by:

- TF-IDF Vectorization
- Cosine Similarity
- NLP text preprocessing

The system analyzes movie descriptions and genres to find similar movies based on textual content.

---

## Future Improvements

- Add collaborative filtering recommendation system
- Integrate TMDB API
- Add movie trailers
- Add user authentication
- Create personal watchlist
- Improve recommendation explainability
- Add sentiment analysis on reviews

---

## Author

Agnes Jeni Makay

Built using Python, NLP, and Streamlit.