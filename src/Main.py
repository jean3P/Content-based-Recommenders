# AUTHOR: Jean 2P. Principe

import numpy as np
import pandas as pd

# read CSV
movies = pd.read_csv('../resources/movies.csv')

# Users ratings. 1 means like, -1 dislike, 0 not rated
john_likes = pd.DataFrame([1, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0])
joan_likes = pd.DataFrame([-1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0])

# Adding the likes to the Dataset
movies["John_Likes"] = john_likes
movies["Joan_Likes"] = joan_likes

# The dataframe with the dataset
print(movies)

# Copy of the original datased  useful for Task 2
movies_weighted = movies.copy()

get_movies_by_genre = movies.iloc[:, 1:11]

# user profile by movie genre
john_likes_score = pd.DataFrame((get_movies_by_genre.values * john_likes.values).sum(axis=0)).T
print(john_likes_score)

joan_likes_score = pd.DataFrame((get_movies_by_genre.values * joan_likes.values).sum(axis=0)).T
print(joan_likes_score)

# Prediction values for  user John
pred_john = (get_movies_by_genre.values * john_likes_score.values).sum(axis=1)
movies["Pred_John"] = pred_john

# Showing the prediction scores for John and the names of the movies
result_prediction_score_john = movies[['Movie', 'John_Likes', 'Pred_John']]
print(result_prediction_score_john)

# Prediction values for user Joan
pred_joan = (get_movies_by_genre.values * joan_likes_score.values).sum(axis=1)
movies["Pred_Joan"] = pred_joan

# Showing the prediction scores for Joan and the names of the movies
result_prediction_score_joan = movies[['Movie', 'Joan_Likes', 'Pred_Joan']]
print(result_prediction_score_joan)
# Which movie does the simple profile predict John will like the most, excluding the movies he has already rated?

excluding_movies_already_rated_by_john = result_prediction_score_john.values[(result_prediction_score_john['John_Likes'] == 0) & (result_prediction_score_john['Pred_John'] > 0)]
movies_dislike_for_joan = result_prediction_score_joan.values[result_prediction_score_joan['Pred_Joan'] < 0]
excluding_movies_already_rated_by_joan = result_prediction_score_joan.values[(result_prediction_score_joan['Joan_Likes'] == 0) & (result_prediction_score_joan['Pred_Joan'] > 0)]

new_movies = movies.iloc[:, 0:12]
get_movies_weighted_by_genre = new_movies.iloc[:, 1:11]
square_root = np.sqrt(new_movies.iloc[:, -1:])
# Adding the likes to the Dataset
new_movies["John_Likes"] = john_likes
new_movies["Joan_Likes"] = joan_likes


movies_weighted = pd.DataFrame(get_movies_weighted_by_genre.values / square_root.values)

# user profile by movie genre
john_likes_score_w = pd.DataFrame((movies_weighted.values * john_likes.values).sum(axis=0)).T
print(john_likes_score_w)

joan_likes_score_w = pd.DataFrame((movies_weighted.values * joan_likes.values).sum(axis=0)).T
print(joan_likes_score_w)

# Prediction values for  user John
pred_john = (movies_weighted.values * john_likes_score_w.values).sum(axis=1)
new_movies["Pred_John"] = pred_john

# Showing the prediction scores for John and the names of the movies
result_prediction_score_john_w = new_movies[['Movie', 'John_Likes', 'Pred_John']]
print(result_prediction_score_john_w)

# Prediction values for user Joan
pred_joan = (movies_weighted.values * joan_likes_score_w.values).sum(axis=1)
new_movies["Pred_Joan"] = pred_joan

# Showing the prediction scores for Joan and the names of the movies
result_prediction_score_joan_w = new_movies[['Movie', 'Joan_Likes', 'Pred_Joan']]
print(result_prediction_score_joan_w)

excluding_movies_already_rated_by_john_w = result_prediction_score_john_w.values[(result_prediction_score_john_w['John_Likes'] == 0) & (result_prediction_score_john_w['Pred_John'] > 0)]
movies_dislike_for_joan_w = result_prediction_score_joan_w.values[result_prediction_score_joan_w['Pred_Joan'] < 0]
excluding_movies_already_rated_by_joan_w = result_prediction_score_joan_w.values[(result_prediction_score_joan_w['Joan_Likes'] == 0) & (result_prediction_score_joan_w['Pred_Joan'] > 0)]