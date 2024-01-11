import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
from math import sqrt

# Dataset
movies = pd.read_csv('C:/Users/urale/Desktop/Artificial_Intelligence/ml-latest-small/movies.csv')
ratings = pd.read_csv('C:/Users/urale/Desktop/Artificial_Intelligence/ml-latest-small/ratings.csv')

user_movie_ratings = ratings.pivot_table(index='userId', columns='movieId', values='rating')
user_movie_ratings = user_movie_ratings.fillna(0)
item_similarity = cosine_similarity(user_movie_ratings.T)

# TFID VECTORIZER
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])

# Cosine Similarity Matrix
content_similarity = cosine_similarity(tfidf_matrix)

# Hyprid Method
def hybrid_recommendation(userId, movieId):
    similar_scores = item_similarity[movieId]
    similar_movies = list(enumerate(similar_scores))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

    # Choose top 10
    similar_movies = similar_movies[:10]

    # Finding smilar movies with content based filtering
    content_scores = content_similarity[movieId]
    content_movies = list(enumerate(content_scores))

    # Sort the ratings
    content_movies = sorted(content_movies, key=lambda x: x[1], reverse=True)

    # Top10
    content_movies = content_movies[:10]

    # Bring together two methods
    hybrid_scores = [(movie[0], 0.5 * similar_scores[movie[0]] + 0.5 * content_scores[movie[0]]) for movie in similar_movies]

    # Sort the ratings
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)

    # Create a reccomendation list
    recommendations = [movie[0] for movie in hybrid_scores]

    return recommendations

def calculate_rmse(predictions, targets):
    mse = mean_squared_error(predictions, targets)
    rmse = sqrt(mse)
    return rmse

# Test user and movie
userId = 3
movieId = 3
recommendations = hybrid_recommendation(userId, movieId)

# Real user ratings
user_ratings = user_movie_ratings.loc[userId].values

# Collaborative Filtering Ratings
collaborative_ratings = user_movie_ratings.T.values[movieId - 1]

# Content Based Filtering Ratings
content_ratings = content_similarity[movieId - 1]

min_length = min(len(collaborative_ratings), len(content_ratings), len(user_ratings))
collaborative_ratings = collaborative_ratings[:min_length]
content_ratings = content_ratings[:min_length]
user_ratings = user_ratings[:min_length]

# Hybrid Recommend Ratings
hybrid_ratings = [0.5 * collaborative_ratings[i] + 0.5 * content_ratings[i] for i in range(len(collaborative_ratings))]

# Calculate RMSE
rmse = calculate_rmse(hybrid_ratings, user_ratings)
print(f"RMSE for Hybrid Recommendation: {rmse}")

# Accuracy precision recall
binary_user_ratings = [1 if rating > 0 else 0 for rating in user_ratings]
binary_hybrid_ratings = [1 if rating > 0 else 0 for rating in hybrid_ratings]

accuracy = accuracy_score(binary_user_ratings, binary_hybrid_ratings)
precision = precision_score(binary_user_ratings, binary_hybrid_ratings)
recall = recall_score(binary_user_ratings, binary_hybrid_ratings)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Shows recommended movie
recommended_movies = movies[movies['movieId'].isin(recommendations)]
print(recommended_movies[['movieId', 'title']])
