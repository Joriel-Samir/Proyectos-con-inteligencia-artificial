from preproces import load_data, read_data, load_dataset
from model import train_model, evaluate_model

movie_path = "C:/Users/jorie/Downloads/ml-25m/movies.csv"
rating_path = "C:/Users/jorie/Downloads/ml-25m/ratings.csv"

movies, ratings = load_data(movie_path, rating_path)
ratings_sample = ratings.sample(frac=0.1, random_state=1)
reader = read_data(ratings_sample)
data = load_dataset(ratings_sample, reader)

model, test_set = train_model(data)
predictions = evaluate_model(model, test_set)

for prediction in predictions[:10]:
    print("This are predictions: ") 
    print(prediction)