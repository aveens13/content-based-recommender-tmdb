import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, JSON, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from ast import literal_eval
import numpy as np

# Define database location
DATABASE_URI = 'sqlite:///movies.db'

# Create a new database connection
engine = create_engine(DATABASE_URI)
Base = declarative_base()

# Define Movies table
class Movie(Base):
    __tablename__ = 'movies'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    budget = Column(Float)
    genres = Column(JSON)
    homepage = Column(String)
    keywords = Column(JSON)
    original_language = Column(String)
    original_title = Column(String)
    overview = Column(Text)
    popularity = Column(Float)
    production_companies = Column(JSON)
    production_countries = Column(JSON)
    release_date = Column(Date)
    revenue = Column(Float)
    runtime = Column(Float)
    spoken_languages = Column(JSON)
    status = Column(String)
    tagline = Column(String)
    vote_average = Column(Float)
    vote_count = Column(Integer)

# Define Credits table
class Credit(Base):
    __tablename__ = 'credits'
    movie_id = Column(Integer, primary_key=True)
    title = Column(String)
    cast = Column(JSON)
    crew = Column(JSON)

# Create tables
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Load data from CSV
movies_df = pd.read_csv('./TMDBDataset/tmdb_5000_movies.csv')
credits_df = pd.read_csv('./TMDBDataset/tmdb_5000_credits.csv')

# Fill NaN values with appropriate defaults
movies_df.fillna({
    'budget': 0.0,
    'genres': '[]',
    'homepage': '',
    'keywords': '[]',
    'original_language': '',
    'original_title': '',
    'overview': '',
    'popularity': 0.0,
    'production_companies': '[]',
    'production_countries': '[]',
    'release_date': '1970-01-01',
    'revenue': 0.0,
    'runtime': 0.0,
    'spoken_languages': '[]',
    'status': '',
    'tagline': '',
    'vote_average': 0.0,
    'vote_count': 0
}, inplace=True)

# Process and insert movies data
for index, row in movies_df.iterrows():
    movie = Movie(
        id=row['id'],
        title=row['title'],
        budget=row['budget'],
        genres=literal_eval(row['genres']),
        homepage=row['homepage'],
        keywords=literal_eval(row['keywords']),
        original_language=row['original_language'],
        original_title=row['original_title'],
        overview=row['overview'],
        popularity=row['popularity'],
        production_companies=literal_eval(row['production_companies']),
        production_countries=literal_eval(row['production_countries']),
        release_date=pd.to_datetime(row['release_date']),
        revenue=row['revenue'],
        runtime=row['runtime'],
        spoken_languages=literal_eval(row['spoken_languages']),
        status=row['status'],
        tagline=row['tagline'],
        vote_average=row['vote_average'],
        vote_count=row['vote_count']
    )
    session.add(movie)

# Fill NaN values in credits data
credits_df.fillna({
    'cast': '[]',
    'crew': '[]'
}, inplace=True)

# Process and insert credits data
for index, row in credits_df.iterrows():
    credit = Credit(
        movie_id=row['movie_id'],
        title=row['title'],
        cast=literal_eval(row['cast']),
        crew=literal_eval(row['crew'])
    )
    session.add(credit)

# Commit the session
session.commit()

# Close the session
session.close()
