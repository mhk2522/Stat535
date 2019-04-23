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
    ''' Gets the overall average movie rating'''
    mu = mat.loc[:,"rating"].mean()
    return mu

def get_movie_mean(i,mu,mat):
    '''Gets the average rating of movie index i and then subtracts mu'''
    movloc = mat.loc[i]["movieID"]
    mov = mat[mat['movieID']==movloc]
    movmean=mov.mean()
    return movmean['rating']-mu

def get_user_mean(u,mu, mat):
    '''Gets the average rating of user index u and then subtracts mu'''
    userloc = mat.loc[u]["userID"]
    user = mat[mat['userID']==userloc]
    usermean=user.mean()
    return usermean['rating']-mu

def get_bias(u,i, mat,mu):   
    ''' Returns the bias of movie i and user j'''
    bui= mu+get_user_mean(u,mu, mat)+get_movie_mean(i,mu,mat)
    return bui
def user_avg_vector(mat,users):
    mu=get_mu(mat)
    uss=np.zeros((1,2353))
    for u in range(2353):
        userloc = users.loc[u]["userID"]
        user = mat[mat['userID']==userloc]
        usermean=user.mean()
        uss[0,u]=usermean['rating']-mu    
    return uss

def movie_avg_vector(mat,movies):
    mu=get_mu(mat)
    movavg=np.zeros((1,1465))
    for i in range(1465):
        movloc = movies.loc[i]["movieID"]
        mov= mat[mat['movieID']==movloc]
        movmean=mov.mean()
        movavg[0,i]=movmean['rating']-mu    
    return movavg

def get_pred_matrix(qtp,mat,users,movies):
    mu=get_mu(mat)
    movvec=np.transpose(movie_avg_vector(mat,movies))
    usevec=user_avg_vector(mat,users)
    nu=np.ones((1465,2353))
    mo=np.ones((1,2353))
    us=np.ones((1465,1))
    mea=mu*nu
    movbias=np.dot(movvec,mo)
    usebias=np.dot(us,usevec)
    pred=mea+movbias+usebias+qtp
    return pred
    
def rating_error(r, q, p,u,i,mu,mat):
    return r - get_bias(u,i,mat,mu)-np.dot(q,p)

def update_q(q, e, p, gam, lam):
    return q + gam * (e * p - lam * q)

def update_p(q, e, p, gam, lam):
    return p + gam * (e * q - lam * p) 

def get_bias_row(u_id,mat):
    mu=get_mu(mat)
    r_bias=np.zeros((1,1465))
    u = mat[mat["userID"]==u_id].index[0]
    for i in range(1465):
            r_bias[0,i]= get_bias(u,i, mat,mu)
    return r_bias
def get_rating(mat,latentmat, u_id, i_id):
    mu=get_mu(mat)
    u = mat[mat["userID"]==u_id].index[0]
    i= mat[mat["movieID"]==i_id].index[0]
    r_bias=get_bias(u,i,mat,mu)
    qpiu=latentmat[i,u]
    return r_bias+qpiu
def get_rating_row(mat,latentmat, u_id):
    u = mat[mat["userID"]==u_id].index[0]
    r_bias=get_bias_row(u_id, mat)
    qpui=np.transpose(latentmat[:,u])
    return r_bias+qpui

def get_top10(row,movies):
    e=np.argsort(row)
    f=[e[0,1455],e[0,1456],e[0,1457],e[0,1458],e[0,1459],e[0,1460],e[0,1461],e[0,1462],e[0,1463],e[0,1464]]
    g=[]
    h=[]
    for i in range(10):
        g.append(movies.loc[f[i]]["movieID"])
    for i in range(10):
       c=movies[movies["movieID"]==g[i]].iloc[0]["name"]
       h.append(c)
    return h
            
def rating_sgd(mat, gam, lam, userID, movieID):
    p = np.ones((4, 2353))*0.1
    q = np.ones((4, 1465))*0.1
    mu = get_mu(mat)
    
    for s in range(5000):
        idx = np.random.choice(len(ratings))
        r = int(mat[['rating']].iloc[idx])
        u_id = int(mat[['userID']].iloc[idx])
        i_id = int(mat[['movieID']].iloc[idx])
        
        u = userID[userID==u_id].index[0]
        i = movieID[movieID==i_id].index[0]

        e = rating_error(r, q[:,i], p[:,u],u_id,i_id,mu,mat)
        # updates p and q till error is <0.01
        while(abs(e)>=0.15):
            q[:,i] = update_q(q[:,i], e, p[:,u], gam, lam)
            p[:,u] = update_p(q[:,i], e, p[:,u], gam, lam)
            e = rating_error(r, q[:,i], p[:,u],u_id,i_id,mu,mat)
    return p,q


random.seed(1)

a=rating_sgd(ratings, 0.1, 0.05, 
           users[['userID']].iloc[:,0], 
           movies[['movieID']].iloc[:,0])

qtp = np.dot(np.transpose(a[1]),a[0])

prediction=get_rating(ratings,qtp,747,1193)

pred_row=get_rating_row(ratings,qtp,747)

top10=get_top10(pred_row,movies)
prediction=get_pred_matrix(qtp,ratings,users,movies)
