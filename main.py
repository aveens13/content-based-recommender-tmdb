from typing import Union

# import contentbasedrecommender as recommender
from fastapi import FastAPI

app = FastAPI()

# popular_movies_data = recommender.popular_movies_list()
# top_rated_movies_data = recommender.top_rated_movies()


@app.get("/")
def read_root():
    return {
        "/popular": "Returns Popular Movies",
        "/toprated": "Returns top rated movies according to imdb weighted rating",
    }


# @app.get("/popular")
# def popular_movies():
#     return popular_movies_data


# @app.get("/toprated")
# def top_rated_movies():
#     return top_rated_movies_data


# @app.get("/recommendation/{movie_title}")
# def return_recommendations(movie_title: Union[str, None]):
#     return recommender.get_recommendations_features(movie_title)
