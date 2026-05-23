# Netflix Movie Recommendation App

A modern movie recommendation system built using Python, Streamlit, Scikit-learn, and Natural Language Processing (NLP).

This application recommends similar movies based on storyline, genre, and content similarity using TF-IDF vectorization and cosine similarity.

---

## Features

- Netflix-inspired user interface
- Movie recommendation system using NLP
- Search movies by title
- Filter movies by:
  - Genre
  - Rating
  - Language
- Trending movies section
- Top rated movies section
- Analytics dashboard
- Movie poster display
- Responsive dark theme interface

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