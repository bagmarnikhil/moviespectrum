from handlers.DatasetHandler import DatasetHandler
from recommenders.ContentBasedRecommender import ContentBasedRecommender

path = "data/ml-latest-small/"
datasetHandler = DatasetHandler(path)
user_ratings = datasetHandler.load_users_ratings()
recommender = ContentBasedRecommender(datasetHandler)
user_profile = recommender.create_user_profile(user_ratings[1])
recommender.present_user_profile(user_profile)
top = recommender.top(user_profile, topN=5)
recommender.present_recommendations(top)
