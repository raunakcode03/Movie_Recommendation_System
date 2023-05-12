#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


credits=pd.read_csv("tmdb_5000_credits.csv")
movies=pd.read_csv("tmdb_5000_movies.csv")


# In[5]:


movies.head()


# In[6]:


credits.columns = ['id','tittle','cast','crew']
movies= movies.merge(credits,on='id')


# In[8]:


movies.head()


# In[9]:


C= movies['vote_average'].mean()
C


# In[10]:


m= movies['vote_count'].quantile(0.9)
m


# In[11]:


q_movies = movies.copy().loc[movies['vote_count'] >= m]
q_movies.shape


# In[12]:


def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[13]:


q_movies['score'] = q_movies.apply(weighted_rating, axis=1)


# In[14]:


q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
q_movies[['title']].head(20)


# In[15]:


import pickle
pickle.dump(q_movies.to_dict(),open('q_movies.pkl','wb'))

