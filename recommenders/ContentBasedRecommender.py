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
                # weights=np.array(user_ratings.values()),
                weights=np.fromiter(user_ratings.values(), dtype=float),
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

    def _cosineKNN_movies_subset(self, movies_subset, movieId, k):
        nbrs = NearestNeighbors(k, metric='cosine', algorithm='brute')
        movies_with_ids = np.array([
            np.hstack(
                [[watched_movie], self.movies_vectors[self.dataset_handler.id2index(watched_movie)]])
            for watched_movie in movies_subset
        ])
        nbrs.fit(movies_with_ids[:, 1:])
        return movies_with_ids[
            nbrs.kneighbors(
                np.array([self.movies_vectors[self.dataset_handler.id2index(movieId)]]), return_distance=False
            )[0],
            0
        ]
    # end

    def predict_rating(self, user_profile, movieId):
        nearest_watched_movies = self._cosineKNN_movies_subset(
            user_profile[1].keys(), movieId, 5)
        return np.average(np.array([user_profile[1][movie] for movie in nearest_watched_movies]))
    # end

    def present_user_profile(self, user_profile):
        print("User favourite genre:",
              self.dataset_handler.feature_index2genre(
                  np.argmax(user_profile[0]))
              )
        print("User ratings:")
        for (movieId, rating) in user_profile[1].items():
            movie_vector = self.movies_vectors[self.dataset_handler.id2index(
                movieId)]
            str_frmt = '{} {}: {}'
            print(str_frmt.format(
                self.dataset_handler.id_to_title[movieId],
                self.dataset_handler.movie_vector2genres(movie_vector),
                rating
            ))
    # end

    def present_recommendations(self, recommendations):
        print("Recommended movies:")
        for movieId in recommendations:
            movie_vector = self.movies_vectors[self.dataset_handler.id2index(
                movieId)]
            print("{} {}".format(
                self.dataset_handler.id_to_title[movieId],
                self.dataset_handler.movie_vector2genres(movie_vector)
            ))
    # end

    def top(self, user_profile, topN):
        return self._cosineKNN_all_movies(user_profile[0], topN)
    # end
