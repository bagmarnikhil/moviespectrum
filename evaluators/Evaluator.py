class Evaluator(object):

    def __init__(self, recommender):
        self.recommender = recommender
    #end

    def computeRMSE(self):
        k_cross = 5
        root_mean_square = 0.0
        total = 0

        users_rating = self.recommender.dataset_handler.load_users_ratings()