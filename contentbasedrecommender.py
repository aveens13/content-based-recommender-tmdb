import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


dataset = pd.read_csv("./TMDBDataset/tmdb_5000_movies.csv")
credits = pd.read_csv("./TMDBDataset/tmdb_5000_credits.csv")

credits.columns = ["id", "title", "cast", "crew"]
dataset = dataset.merge(credits, on="id")

mean_votes = dataset["vote_average"].mean()
minimum_votes_required = dataset["vote_count"].quantile(0.9)  # minimum vote counts

movies_list = dataset.copy().loc[dataset["vote_count"] >= minimum_votes_required]


def weighted_rating(x, m=minimum_votes_required, c=mean_votes):
    v = x["vote_count"]
    r = x["vote_average"]
    # IMDB formula
    return (r * v + c * m) / (v + m)


movies_list["score"] = movies_list.apply(weighted_rating, axis=1)
movies_list = movies_list.sort_values("score", ascending=False)

popular_movies = dataset.sort_values("popularity", ascending=False)

popular_movies["genres"] = popular_movies["genres"].apply(literal_eval)
movies_list["genres"] = movies_list["genres"].apply(literal_eval)


def popular_movies_list():
    return (
        popular_movies[["id", "title_x", "genres"]].head(10).to_dict(orient="records")
    )


def top_rated_movies():
    return (
        movies_list[["id", "title_x", "genres", "score"]]
        .head(10)
        .to_dict(orient="records")
    )


top_budget_movies = dataset.sort_values("budget", ascending=False)


# TFIDF
tfidf = TfidfVectorizer(
    stop_words="english"
)  # Defining a TF-IDF vectorizer and removing the stop words in english

# now replacing the empty NaN in overview with empty strings
dataset["overview"] = dataset["overview"].fillna("")

tfidf_matrix = tfidf.fit_transform(dataset["overview"])

cosine_sim = linear_kernel(
    tfidf_matrix, tfidf_matrix
)  # Caculating the cosine similarties between any two overview in the dataset

indices = pd.Series(dataset.index, index=dataset["id"])


def get_recommendations_overview(id, cosine_sim=cosine_sim):
    idx = indices[id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return dataset[["id", "title_x"]].iloc[movie_indices].to_dict(orient="records")


# Recommender using keywords as well as cast director etc
# Cleaining the data and creating a soup for all the features combined

features = ["cast", "crew", "genres", "keywords"]
# changing the string representations of different columns in dataset into the correspoinding python lists or objects
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
        if len(names) > 3:
            names = names[:3]
        return names

    return []


dataset["director"] = dataset["crew"].apply(get_director)
features = ["cast", "keywords", "genres"]

for feature in features:
    dataset[feature] = dataset[feature].apply(get_list)


# Cleaning the data if it conains any spaces in middle than removing the spaces
def clean_data(x):
    if isinstance(x, list):
        return [i.lower().replace(" ", "") for i in x]
    else:
        if isinstance(x, str):
            return x.lower().replace(" ", "")
        else:
            return ""


features = ["cast", "keywords", "director", "genres"]

for feature in features:
    dataset[feature] = dataset[feature].apply(clean_data)

# Joining the features and creating the soup


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
dataset = dataset.reset_index()
indices2 = pd.Series(dataset.index, index=dataset["id"])


def get_recommendations_features(id, cosine_sim=cosine_sim2):
    if id not in indices2:
        return []
    idx = indices2[id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:5]
    movie_indices = [i[0] for i in sim_scores]
    return dataset[["id", "title_x"]].iloc[movie_indices].to_dict(orient="records")
