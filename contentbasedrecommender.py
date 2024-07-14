import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval
from functools import lru_cache


# Lazy loading of data
@lru_cache(maxsize=1)
def load_data():
    dataset = pd.read_csv("./TMDBDataset/tmdb_5000_movies.csv")
    credits = pd.read_csv("./TMDBDataset/tmdb_5000_credits.csv")
    credits.columns = ["id", "title", "cast", "crew"]
    dataset = dataset.merge(credits, on="id")
    return dataset


# Function to get popular movies
@lru_cache(maxsize=1)
def popular_movies_list():
    dataset = load_data()
    popular_movies = dataset.sort_values("popularity", ascending=False)
    popular_movies["genres"] = popular_movies["genres"].apply(literal_eval)
    return (
        popular_movies[["id", "title_x", "genres"]].head(10).to_dict(orient="records")
    )


# Function to get top rated movies
@lru_cache(maxsize=1)
def top_rated_movies():
    dataset = load_data()
    mean_votes = dataset["vote_average"].mean()
    minimum_votes_required = dataset["vote_count"].quantile(0.9)

    def weighted_rating(x, m=minimum_votes_required, c=mean_votes):
        v = x["vote_count"]
        r = x["vote_average"]
        return (r * v + c * m) / (v + m)

    movies_list = dataset[dataset["vote_count"] >= minimum_votes_required].copy()
    movies_list["score"] = movies_list.apply(weighted_rating, axis=1)
    movies_list = movies_list.sort_values("score", ascending=False)
    movies_list["genres"] = movies_list["genres"].apply(literal_eval)

    return (
        movies_list[["id", "title_x", "genres", "score"]]
        .head(10)
        .to_dict(orient="records")
    )


# TF-IDF and cosine similarity calculation
@lru_cache(maxsize=1)
def calculate_tfidf_similarity():
    dataset = load_data()
    tfidf = TfidfVectorizer(stop_words="english")
    dataset["overview"] = dataset["overview"].fillna("")
    tfidf_matrix = tfidf.fit_transform(dataset["overview"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim


# Get recommendations based on overview
def get_recommendations_overview(id):
    dataset = load_data()
    cosine_sim = calculate_tfidf_similarity()
    indices = pd.Series(dataset.index, index=dataset["id"])
    idx = indices[id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return dataset[["id", "title_x"]].iloc[movie_indices].to_dict(orient="records")


# Feature-based similarity calculation
@lru_cache(maxsize=1)
def calculate_feature_similarity():
    dataset = load_data()
    features = ["cast", "crew", "genres", "keywords"]
    for feature in features:
        dataset[feature] = dataset[feature].apply(literal_eval)

    def get_director(x):
        for i in x:
            if i["job"] == "Director":
                return i["name"]
        return np.nan

    def get_list(x):
        if isinstance(x, list):
            names = [i["name"] for i in x]
            return names[:3] if len(names) > 3 else names
        return []

    dataset["director"] = dataset["crew"].apply(get_director)
    features = ["cast", "keywords", "genres"]
    for feature in features:
        dataset[feature] = dataset[feature].apply(get_list)

    def clean_data(x):
        if isinstance(x, list):
            return [i.lower().replace(" ", "") for i in x]
        elif isinstance(x, str):
            return x.lower().replace(" ", "")
        else:
            return ""

    features = ["cast", "keywords", "director", "genres"]
    for feature in features:
        dataset[feature] = dataset[feature].apply(clean_data)

    def create_soup(x):
        return (
            " ".join(x["keywords"])
            + " "
            + " ".join(x["cast"])
            + " "
            + x["director"]
            + " "
            + " ".join(x["genres"])
        )

    dataset["soup"] = dataset.apply(create_soup, axis=1)
    count = CountVectorizer(stop_words="english")
    count_matrix = count.fit_transform(dataset["soup"])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    return cosine_sim2, dataset


# Get recommendations based on features
def get_recommendations_features(id):
    cosine_sim2, dataset = calculate_feature_similarity()
    indices2 = pd.Series(dataset.index, index=dataset["id"])
    if id not in indices2:
        return []
    idx = indices2[id]
    sim_scores = list(enumerate(cosine_sim2[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:5]
    movie_indices = [i[0] for i in sim_scores]
    return dataset[["id", "title_x"]].iloc[movie_indices].to_dict(orient="records")
