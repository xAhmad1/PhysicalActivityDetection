from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


class FeatureExtractor(BaseEstimator):

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X



def get_estimator():

    feature_extractor = FeatureExtractor()

    classifier1 = LogisticRegression(multi_class='multinomial', solver='lbfgs')
	#extra comment
    pipe = make_pipeline(feature_extractor, classifier1)
    return pipe