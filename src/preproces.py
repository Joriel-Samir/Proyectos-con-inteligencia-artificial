import pandas as pd
from surprise import Reader, Dataset

movie_path = "C:/Users/jorie/Downloads/ml-25m/movies.csv"
rating_path = "C:/Users/jorie/Downloads/ml-25m/ratings.csv"

def load_data(movie_path, rating_path):
    movies = pd.read_csv(movie_path)
    ratings = pd.read_csv(rating_path)
    
    print("Movies Data: ")
    print(movies.head())
    print("Ratings Data: ")
    print(ratings.head())
    return movies, ratings

def read_data(ratings):
    min_rating = ratings["rating"].min()
    max_rating = ratings["rating"].max()
    reader = Reader(rating_scale=(min_rating, max_rating))
    return reader

def load_dataset(ratings, reader): 
    data = Dataset.load_from_df(ratings[["userId","movieId","movieId"]], reader)
    return data