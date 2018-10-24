import os
import pandas as pd
import numpy as np

genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western"
]


class DatasetHandler(object):

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
    # end

    def load_movies(self):
        self.id_to_title = {}
        self.movie_index_to_movie_id = []
        movies_vectors = []
        movies_frame = pd.read_csv(
            os.path.join(self.dataset_path, "movies.csv"), names=["movieId", "title", "genres"]
        )
        for _, row in movies_frame.iterrows():
            genres_list = row["genres"].split("|")
            self.id_to_title[int(row["movieId"])] = row["title"]
            self.movie_index_to_movie_id.append(int(row["movieId"]))
            movies_vectors.append(
                np.array([1 if genre in genres_list else 0 for genre in genres])
            )
        return np.array(movies_vectors)
    # end

    def load_users_ratings(self):
        users_ratings = {}
        ratings_frame = pd.read_csv(
            os.path.join(self.dataset_path, "ratings.csv"), names=["userId", "movieId", "rating", "timestamp"]
        )
        for _, row in ratings_frame.iterrows():
            userId = int(row["userId"])
            movieId = int(row["movieId"])
            if userId not in users_ratings:
                users_ratings[userId] = {}
            users_ratings[userId][movieId] = row["rating"]
        return users_ratings
    # end

    def id2index(self, movieId):
        return self.movie_index_to_movie_id.index(movieId)
    # end

    def indices2ids(self, indices):
        return [self.movie_index_to_movie_id[index] for index in indices]
    # end


'''
Driver program to test

path = "../data/ml-latest-small/"
datasetHandler = DatasetHandler(path)
datasetHandler.load_movies()
datasetHandler.load_users_ratings()

'''
