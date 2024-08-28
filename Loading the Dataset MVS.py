import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise.model_selection import cross_validate

# Load MovieLens 100k dataset
movies = pd.read_csv('u.item', sep='|', header=None, encoding='latin-1', usecols=[0, 1], names=['movie_id', 'title'])
ratings = pd.read_csv('u.data', sep='	', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Merge ratings with movies to get movie titles
data = pd.merge(ratings, movies, on='movie_id')
data.head()
