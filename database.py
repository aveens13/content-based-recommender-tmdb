import sqlite3
import pandas as pd

# Paths to the CSV files
credits_csv_path = "./TMDBDataset/tmdb_5000_movies.csv"
movies_csv_path = "./TMDBDataset/tmdb_5000_credits.csv"

# Load the CSV files into pandas DataFrames
credits_df = pd.read_csv(credits_csv_path)
movies_df = pd.read_csv(movies_csv_path)

# Path to the SQLite database file
sqlite_db_path = "./movies.db"

# Connect to the SQLite database (it will be created if it doesn't exist)
conn = sqlite3.connect(sqlite_db_path)

# Save the DataFrames to the SQLite database
credits_df.to_sql("tmdb_credits", conn, if_exists="replace", index=False)
movies_df.to_sql("tmdb_movies", conn, if_exists="replace", index=False)

# Close the connection
conn.close()

print(f"Data has been successfully saved to {sqlite_db_path}")
