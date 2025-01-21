

from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
def train_model(data):
    train_set, test_set = train_test_split(data,test_size=0.25)
    model = SVD()
    model.fit(train_set)
    return model, test_set

def evaluate_model(model, test_set):
    predictions = model.test(test_set)
    rmse = accuracy.rmse(predictions)
    print(f"RMSE: {rmse}")
    return predictions