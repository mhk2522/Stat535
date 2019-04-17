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

plt.hist(allData['rating'], color = 'blue', edgecolor = 'black', bins = 10)

moviesfin=pd.merge(ratings_count, avgratings, on='movieID')
moviesfin.head()
matrix_A=pd.DataFrame(allData.pivot_table(index="movieID", columns="userID", values='rating'))
matrix_A=matrix_A.fillna(0)
matrix_A.head()
matrix_B=pd.DataFrame(allData.pivot_table(index="userID", columns="movieID", values='rating'))
matrix_B.head()


from scipy.sparse import csr_matrix
# pivot ratings into movie features
df_movie_matrix = allData.pivot(
    index='userID',
    columns='movieID',
    values='rating'
)
# convert dataframe of movie features to scipy sparse matrix
mat_movie_matrix = np.array(df_movie_matrix.values)
random.seed(1)
def rate_predict(R,p,q,k,lamb, gamma):
    
    for i in range(len(R)):
        for j in range(len(R[0])):
            if R[i,j]>0:
                q_t=np.transpose(q)
                error=R[i,j]-np.dot(p[i,:],q_t[:,j])
                for l in range(k):
                    q[j,l]=q[j,l]+gamma*(error*p[i,l]-lamb*q[j,l])
                    p[i,l]=p[i,l]+gamma*(error*q[j,l]-lamb*(p[i,l]))
                    
    return p,q_t
P=np.random.rand(len(mat_movie_matrix),2)
Q=np.random.rand(len(mat_movie_matrix[0,:]),2)
beta=rate_predict(mat_movie_matrix,P,Q,2,0,0.1)
Answer=np.dot(beta[0],beta[1])

def rating_error(r, q, p):
    return r - np.dot(q,p)

def update_q(q, e, p, gam, lam):
    return q + gam * (e * p - lam * q)

def update_p(q, e, p, gam, lam):
    return p + gam * (e * q - lam * p)

def rating_sgd(ratings, gam, lam, userID, movieID):
    p = np.ones((4, 2353))*0.1
    q = np.ones((4, 1465))*0.1
    
    for s in range(50000):
        idx = np.random.choice(len(ratings))
        r = int(ratings[['rating']].iloc[idx])
        u_id = int(ratings[['userID']].iloc[idx])
        i_id = int(ratings[['movieID']].iloc[idx])
        
        u = userID[userID==u_id].index[0]
        i = movieID[movieID==i_id].index[0]
        
        e = rating_error(r, q[:,i], p[:,u])
        q[:,i] = update_q(q[:,i], e, p[:,u], gam, lam)
        p[:,u] = update_p(q[:,i], e, p[:,u], gam, lam)
        
    return p,q

def predict_rating(p, q):
    return np.dot(np.transpose(q),p)
        
a=rating_sgd(ratings, 0.1, 0, 
           users[['userID']].iloc[:,0], 
           movies[['movieID']].iloc[:,0])
b=predict_rating(a[0],a[1])

