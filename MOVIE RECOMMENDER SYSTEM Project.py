#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


credits=pd.read_csv("tmdb_5000_credits.csv")
movies=pd.read_csv("tmdb_5000_movies.csv")


# In[3]:


movies.head()


# In[4]:


#credits.head(1)['cast'].values
credits.head(1)['crew'].values


# In[5]:


movies=movies.merge(credits,on='title')


# In[6]:


movies.head(1)


# In[7]:


#genres
#id
#keywords
#Title
#Overview
#Cast
#Crew
movies=movies[["movie_id",'title','genres','keywords','overview','cast','crew']]


# In[8]:


movies.head()


# In[9]:


movies.isnull().sum()


# In[10]:


movies.dropna(inplace=True)


# In[11]:


movies.duplicated().sum()


# In[12]:


movies.genres.iloc[0]


# In[13]:


ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[14]:


import ast
def convert(obj):
    l1=[]
    for i in ast.literal_eval(obj):
        l1.append(i['name'])
    return l1


# In[15]:


movies['genres']=movies['genres'].apply(convert)


# In[16]:


movies.head()


# In[17]:


movies['keywords']=movies['keywords'].apply(convert)


# In[18]:


movies.head()


# In[19]:


def convert3(obj):
    l1=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter != 3:
            l1.append(i['name'])
            counter=counter+1
        else:
            break
    return l1


# In[20]:


movies['cast']=movies['cast'].apply(convert3)


# In[21]:


movies.head()


# In[22]:


def fetch_director(obj):
    l1=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l1.append(i['name'])
    return l1


# In[23]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[24]:


movies.head()


# In[25]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[26]:


movies.head()


# In[27]:


movies['genres']=movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])


# In[28]:


movies.head()


# In[34]:


title_list=[]
for i in movies[title]:
    i.append(title_list)
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']+title_list


# In[30]:


movies.head()


# In[31]:


new_df=movies[['movie_id','title','tags']]


# In[32]:


new_df['tags']=new_df['tags'].apply (lambda x:" ".join(x))


# In[33]:


new_df


# In[34]:


new_df['tags'][0]


# In[35]:


new_df['tags']=new_df['tags'].apply (lambda x:x.lower())


# In[36]:


new_df['tags'][0]


# In[37]:


new_df


# In[38]:


import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[39]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return" ".join(y)


# In[40]:


new_df['tags']=new_df['tags'].apply(stem)


# In[41]:


new_df['tags'][0]


# In[42]:


new_df['tags'][1]


# In[43]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[44]:


cv.fit_transform(new_df['tags']).toarray().shape


# In[45]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[46]:


vectors[0]


# In[47]:


len(cv.get_feature_names())


# In[48]:


cv.get_feature_names()


# In[49]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)


# In[50]:


similarity


# In[51]:


similarity[0]


# In[52]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances=similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key = lambda x: x[1])[1:6]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)


# In[53]:


recommend('Avatar')


# In[54]:


recommend('Batman Begins')


# In[55]:


import pickle


# In[56]:


pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))


# In[57]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




