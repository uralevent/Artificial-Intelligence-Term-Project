import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate

# Import and read the dataset
file_path = 'C:/Users/urale/Desktop/Recommander/ml-latest-small/ml-latest-small/ratings.csv'
df = pd.read_csv(file_path)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# User-Based Recommender System
sim_options_user = {'name': 'cosine', 'user_based': True}
user_based_algo = KNNBasic(sim_options=sim_options_user)

# Item-Based Recommender System
sim_options_item = {'name': 'cosine', 'user_based': False}
item_based_algo = KNNBasic(sim_options=sim_options_item)

# Comparison Between User-Based and Item-Based Recommender Systems
user_based_results = cross_validate(user_based_algo, data, measures=['RMSE'], cv=5, verbose=True)
item_based_results = cross_validate(item_based_algo, data, measures=['RMSE'], cv=5, verbose=True)

# Write the Results
print("User-Based RMSE: ", user_based_results['test_rmse'].mean())
print("Item-Based RMSE: ", item_based_results['test_rmse'].mean())

