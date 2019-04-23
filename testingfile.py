# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 00:58:25 2019

@author: sabba
"""

import pandas as pd
import numpy as np
import seaborn as sns
#import pdb
import matplotlib.pyplot as plt 
import sklearn.model_selection as model_selection
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import pairwise_distances 
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn import linear_model
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors

users=pd.read_csv("users.csv")
movies=pd.read_csv("movies.tsv", sep="\t")
ratings=pd.read_csv("ratings.csv")
predict=pd.read_csv("predict.csv")
allData=pd.read_csv("allData.tsv", sep="\t")
    
def trainingMatrixGenerator(rating):
    
    """Makes two matrices, one with the movieID and the number of ratings and
    one with the userID and the number of ratings they did. Then we remove 
    rows where the number of ratings<5 to prevent prediction errors. We then take
    20% of the possible ratings as the test set. We take the remaining ratings
    as the test set. The function takes in a dataframe and returns in the
    order------> testset, trainingset"""
    
    ratings_movie_count=pd.DataFrame(allData.groupby('movieID',  as_index=False)['rating'].count())
    ratings_movie_count.rename(index=str, columns={'movieID':'movieID','rating':'count'}, inplace=True)

    ratings_user_count=pd.DataFrame(allData.groupby('userID',  as_index=False)['rating'].count())
    ratings_user_count.rename(index=str, columns={'userID':'userID','rating':'count'}, inplace=True)
    
    #Remove ratings where the count<5 for movies and users
    ratings_user_count = ratings_user_count[ratings_user_count["count"]>5]
    ratings_movie_count = ratings_movie_count[ratings_movie_count["count"]>5]
    
    #Create a copy dataframe for the training set and test set
    trainingset = ratings.copy()
    testset = ratings.copy()
    
    
    for i in range(rating.shape[0]):
        u_id = int(ratings[['userID']].iloc[i])
        i_id = int(ratings[['movieID']].iloc[i])
        
    #Removes possibilites where the number of user/movie ratings<5 by checking count tables
        if u_id not in ratings_user_count.userID.values or i_id not in ratings_movie_count.movieID.values:
            testset = testset[~((testset["userID"]==u_id) & (testset["movieID"]==i_id))]
            
    #Takes a sample of 20% from testset
    testset = testset.sample(frac=0.2)
    
    #remaining_test = test.loc[~test.index.isin(testset.index)]
    #Reindex testset
    testset = testset.reset_index(drop=True)
    
    #Remove rows from trainingset which are in the testset
    for i in range(testset.shape[0]):
        u_id = int(testset[['userID']].iloc[i])
        i_id = int(testset[['movieID']].iloc[i])
        trainingset = trainingset[~((trainingset["userID"]==u_id) & (trainingset["movieID"]==i_id))]
        
    #Reindex the trainingset
    trainingset = trainingset.reset_index(drop=True)
        
    return testset, trainingset