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


def get_mu(mat):
    mu = mat.loc[:,"rating"].mean()
    return mu

def get_movie_mean(i_id,mu,mat):
    mov = mat[mat['movieID']==i_id]
    movmean=mov.mean()
    return movmean['rating']-mu

def get_user_mean(u_id,mu, mat):
    user = mat[mat['userID']==u_id]
    usermean=user.mean()
    return usermean['rating']-mu

def get_bias(u_id,i_id, mat,mu):   
    bui= mu+get_user_mean(u_id,mu, mat)+get_movie_mean(i_id,mu, mat)
    return bui

def rating_error(r, q, p,u_id,i_id,mu,mat):
    return r - get_bias(u_id,i_id,mat,mu)-np.dot(q,p)

def update_q(q, e, p, gam, lam):
    return q + gam * (e * p - lam * q)

def update_p(q, e, p, gam, lam):
    return p + gam * (e * q - lam * p) 

def get_bias_matrix(i,u,u_id,i_id,mat,mu):
    r_bias=np.zeros((1465,2353))
    for i in range(1465):
        for u in range(2353):
            r_bias[i,u]= get_bias(u_id,i_id,mat,mu)
    return r_bias

def rating_sgd(mat, gam, lam, userID, movieID):
    p = np.ones((4, 2353))*0.1
    q = np.ones((4, 1465))*0.1
    mu = get_mu(mat)
    
    for s in range(100):
        idx = np.random.choice(len(ratings))
        r = int(mat[['rating']].iloc[idx])
        u_id = int(mat[['userID']].iloc[idx])
        i_id = int(mat[['movieID']].iloc[idx])
        
        u = userID[userID==u_id].index[0]
        i = movieID[movieID==i_id].index[0]
        r_pred= get_bias_matrix(i,u,u_id,i_id,mat,mu)
        e = rating_error(r, q[:,i], p[:,u],u_id,i_id,mu,mat)
        while(e>=0.01):
            q[:,i] = update_q(q[:,i], e, p[:,u], gam, lam)
            p[:,u] = update_p(q[:,i], e, p[:,u], gam, lam)
            e = rating_error(r, q[:,i], p[:,u],u_id,i_id,mu,mat)
    r_pred = r_pred + np.dot(np.transpose(q),p)

            
            
        
    return r_pred
        
a=rating_sgd(ratings, 0.1, 0, 
           users[['userID']].iloc[:,0], 
           movies[['movieID']].iloc[:,0])
#  Still working on, just update
