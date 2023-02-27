from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import pandas as pd

def compute_moving_average(X_df, feature, time_window, center=False):
    """
    For a given dataframe, compute the moving average over
    a defined period of time (time_window) of a defined feature
 
    Parameters
    ----------
    X : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling std from
    time_window : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    X_cop = X_df.copy()
    name = "_".join([feature, str(time_window), "avg"])
    X_cop[name] = X_cop.groupby('subject_id')[feature].rolling(time_window).mean().reset_index(0,drop=True)
    X_cop[name] = X_cop[name].ffill().bfill()
    X_cop[name] = X_cop[name].astype(X_cop[feature].dtype)

    return X_cop

def compute_lagged_features(X_df, feature, lag_value):
    name  = feature + "_lag_" + str(lag_value)
    X_cop = X_df.copy()
    X_cop[name] = X_cop[feature].shift(lag_value)
    X_cop[name] = X_cop[name].ffill().bfill()
    return X_cop


class FeatureExtractor(BaseEstimator):

    def fit(self, X, y):
        return self

    def transform(self, X):
        X = compute_moving_average(X,'heartrate', 3)
        return X



def get_estimator():

    feature_extractor = FeatureExtractor()

    classifier1 = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    pipe = make_pipeline(feature_extractor, classifier1)
    return pipe