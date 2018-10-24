import numpy as np
from sklearn.neighbors import NearestNeighbors


class ContentBasedRecommender(object):

    def __init__(self, dataset_handler):
        self.dataset_handler = dataset_handler
        self.movies_vectors = dataset_handler.load_movies()
    # end

    def train(self):
        pass
    # end

    def create_user_profile(self, user_ratings):
        return (
            np.average(
                np.array([
                    self.movies_vectors[self.dataset_handler.id2index(movie)]
                    for (movie, rating) in user_ratings.items()
                ]),
                weights=np.array(user_ratings.values()),
                axis=0
            ),
            user_ratings
        )
    # end

    def _cosineKNN_all_movies(self, user_profile, k):
        nbrs = NearestNeighbors(metric='cosine', algorithm='brute')
        nbrs.fit(self.movies_vectors)
        return(
            self.dataset_handler.indices2ids(
                nbrs.kneighbors(np.array([user_profile]),
                                k, return_distance=False)[0]
            )
        )
    # end
