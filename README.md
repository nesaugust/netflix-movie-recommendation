# Netflix Movie Recommendation App 🎬

A movie recommendation system built using Python, NLP, TF-IDF, and Streamlit.

## Features
- Movie recommendation based on content similarity
- NLP text processing using TF-IDF
- Interactive Streamlit web app
- Movie posters, ratings, and overview display

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- TF-IDF Vectorizer
- Cosine Similarity

## Dataset
Movie metadata dataset containing:
- Title
- Overview
- Genre
- Popularity
- Vote Average
- Poster URL

## How It Works
The recommendation system combines movie overview and genre, converts text into TF-IDF vectors, and calculates similarity between movies using cosine similarity.

## Run Locally

```bash
streamlit run app.py
```

## Project Structure

```text
Netflix/
│
├── app.py
├── clean_movies.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Future Improvements
- Add search by genre
- Add filtering by rating
- Deploy online using Streamlit Cloud
- Add collaborative filtering recommendation system

## Author
Agnes Jeni Makay