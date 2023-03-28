#Exercise 3: User-User Filtering
#Authors: Jean 2P. Principe

import numpy as np
import pandas as pd
import math

#SupportFunctions

#This function receives the correlation values and a list of ratings, if  the rating is different than Nan
#the we multiply  the value of the correlations by the rating otherwise no. The sum of all the operations is returned.
def sumproduct(correlations, ratings):
    sum=0
    for i in range(len(list(correlations))):
       if not math.isnan(ratings[i]):
            sum=sum+(correlations[i]*ratings[i])
    return sum

#This function receives the correlation values and a list of ratings, if  the rating is different than Nan
#the we sum up the value of the correlations, otherwise no. The value of the sum is returned
def sumif(correlations, ratings):
    sum=0
    for i in range(len(list(correlations))):
        if not math.isnan(ratings[i]):
            sum=sum+correlations[i]
    return sum

#Implementation of the Not Normalized score rating function
def prediction_not_normalized(correlations,ratings):
    if sumif(correlations.values.reshape(5), ratings)>0:
        return sumproduct(correlations.values.reshape(5), ratings) / sumif(correlations.values.reshape(5), ratings)
    return np.nan


#**********************MAIN PROGRAM*******************************************************************
# read data
movie_ratings = pd.read_csv('../resources/movie-ratings.csv', header=0)
#copying datased for task 2
movie_ratings_weighted=movie_ratings.copy()

#Calculating the pearson correlation among users
user_by_user_corr_matrix=movie_ratings.corr(method='pearson')
# user_by_user_corr_matrix

corr_top5=(user_by_user_corr_matrix.loc['Sofia'].sort_values(ascending=False)[1:6]).to_frame().T
corr_top5_a=(user_by_user_corr_matrix.loc['Arielle'].sort_values(ascending=False)[1:6]).to_frame().T


#This list is basically to select the rating of the top5 users
selection_labels= ['Movie']+ corr_top5.columns.tolist()
print(selection_labels)
#Here we are using the previous list to select the ratings
rating_top5=movie_ratings.loc[:,selection_labels]
print(rating_top5)

#Empty list to store the results of the prediction scores
prediction_results = []

#We iterate over the rows of the top5 similar users ratings
for index, row in rating_top5.iterrows():
    #Getting the rating values for the movies of each user of the top 5
    ratings_row=row[selection_labels[1:]].values
    #Computing the prediction values, using the not normalized model. We call this function sending as parameters
    #the correlation values of the top 5 users and the ratings they have assigned to the items
    pred_value=prediction_not_normalized(corr_top5,ratings_row)
    #List with the results of the prediction, we add a new result in each iteration, one per each item
    prediction_results.append(pred_value)

print(prediction_results)

#Adding a new column to our original DataFrame
movie_ratings['Prediction']=prediction_results

#Visualizing the items sorted from highest prediction score to lowest
#We should recommend the items with the highest prediction score
print(movie_ratings.sort_values(by='Prediction', ascending=False))

arielle_ratings = movie_ratings.loc[:, 'Arielle']