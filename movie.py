# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:08:53 2019

@author: Matthew
"""

import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import pdb
import matplotlib.pyplot as plt 
import sklearn.model_selection as model_selection
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn import linear_model
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
users=pd.read_csv("users.csv")
movies=pd.read_csv("movies.tsv", sep="\t")
ratings=pd.read_csv("ratings.csv")
predict=pd.read_csv("predict.csv")
allData=pd.read_csv("allData.tsv", sep="\t")
allData.head()
movies.head()
users.head()
avgratings=pd.DataFrame(allData.groupby("movieID", as_index=False).agg({"rating":"mean"}))
avgratings.rename(index=str, columns={'movieID':'movieID','rating':'avg_rating'}, inplace=True)
avgratings.head()
ratings_count=pd.DataFrame(allData.groupby('movieID',  as_index=False)['rating'].count())
ratings_count.rename(index=str, columns={'movieID':'movieID','rating':'count'}, inplace=True)
ratings_count.head()
ratings_count['count'].hist(bins=30)
moviesfin=pd.merge(ratings_count, avgratings, on='movieID')
moviesfin.head()
matrix_A=pd.DataFrame(allData.pivot_table(index="movieID", columns="userID", values='rating'))
matrix_A