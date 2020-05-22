# author: Shivam
from unittest.mock import inplace

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import  cosine_similarity

def get_title_from_index(index):
    return data[data.index==index]['title'].values[0]


def get_index_from_title(title):
    return data[data['title']==title]['index'].values[0]

def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
# Reading CSV
data = pd.read_csv('movie_dataset.csv')
print(data.head())

X = data[['keywords','cast','genres','director']]

# print(X.head())

X['keywords'].fillna('',inplace=True)

print(X.info())
#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   keywords  4803 non-null   object
#  1   cast      4760 non-null   object
#  2   genres    4775 non-null   object
#  3   director  4773 non-null   object

X.dropna(inplace=True)

X['combined_feature']=X.apply(combine_features,axis=1)


print(X['combined_feature'].head())


cv = CountVectorizer()

count_matrix = cv.fit_transform(X['combined_feature'])

similarity_among_movies= cosine_similarity(count_matrix)

movie_user_likes = "Avatar"

movie_index= get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(similarity_among_movies[movie_index]))

sorted_similar_movies = sorted(similar_movies,key=lambda x: x[1],reverse=True)

i=0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    if i>50:
        break
    i+=1

# Recommended movies similar to Avatar.

# Guardians of the Galaxy
# Aliens
# Awake
# Star Trek Into Darkness
# Star Trek Beyond
# Red Dog
# Lockout
# I Think I Love My Wife
# Moonraker
# Planet of the Apes
# Galaxy Quest
# Gravity
# Alien³
# Jupiter Ascending
# The Wolverine
# Airborne
# Zathura: A Space Adventure
# Subconscious
# Sexy Beast
# Wing Commander
# Star Trek
# Lost in Space
# Babylon A.D.
# The Fifth Element
# Oblivion
# Titan A.E.
# AVP: Alien vs. Predator
# The Empire Strikes Back
# Dragonball Evolution
# Superman Returns
# Divergent
# John Carter
# The Black Hole
# The Ultimate Gift
# Memoirs of an Invisible Man
# Starship Troopers
# The Astronaut's Wife
# Machete Kills
# Soldier
# The Abyss
# Damnation Alley
# Men in Black
# Space Cowboys
# Space Dogs
# The Time Machine
# Sheena
# Captain America: Civil War
# Star Trek: Insurrection
# Oz: The Great and Powerful
# The One
# X-Men: Days of Future Past
#

