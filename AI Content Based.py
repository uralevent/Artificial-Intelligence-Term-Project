import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import cross_validate

# dataset
movies = pd.read_csv('C:/Users/urale/Desktop/Artificial_Intelligence/ml-latest-small/movies.csv')
ratings = pd.read_csv('C:/Users/urale/Desktop/Artificial_Intelligence/ml-latest-small/ratings.csv')

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# spliting
trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)

# predictions on test set
predictions = model.test(testset)

# rmse
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

# Cross-validation
cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(f"Cross-validation results: {cv_results}")

