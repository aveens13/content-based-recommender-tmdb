from fastapi import FastAPI
import contentbasedrecommender as recommender

app = FastAPI()


@app.get("/")
def read_root():
    return {
        "/popular": "Returns Popular Movies",
        "/toprated": "Returns top rated movies according to imdb weighted rating",
    }


@app.get("/popular")
def popular_movies():
    return recommender.popular_movies_list()


@app.get("/toprated")
def top_rated_movies():
    return recommender.top_rated_movies()


@app.get("/recommendation/{movie_id}")
def return_recommendations(movie_id: int):
    return recommender.get_recommendations_features(movie_id)


@app.get("/recommendation/overview/{movie_id}")
def return_recommendations_using_overview(movie_id: int):
    return recommender.get_recommendations_overview(movie_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
