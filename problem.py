import os
import datetime

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from rampwf.workflows.sklearn_pipeline import SKLearnPipeline
from rampwf.workflows.sklearn_pipeline import Estimator

problem_title = "Physical Activity Prediction"


# -----------------------------------------------------------------------------
# Worklow element
# -----------------------------------------------------------------------------

workflow = rw.workflows.Estimator()


# -----------------------------------------------------------------------------
# Predictions type
# -----------------------------------------------------------------------------
labels = [ 1,  2,  3,  4,  5,  6,  7, 12, 13, 16, 17, 24]
Predictions = rw.prediction_types.make_multiclass(label_names=labels)


# -----------------------------------------------------------------------------
# Score types
# -----------------------------------------------------------------------------

class custom_acc(BaseScoreType): # otherwise smoothing won't affect accuracy
    is_lower_the_better = False

    def __init__(self, name='acc', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        class_dict = dict(enumerate(labels))
        y_pred1 = (y_pred.argmax(axis=1))
        y_true = (y_true.argmax(axis=1))
        y_pred_names = [class_dict[i] for i in y_pred1]
        y_true = [class_dict[i] for i in y_true]
        score = accuracy_score(y_true,y_pred_names)
        return score
    
score_types = [
    rw.score_types.NegativeLogLikelihood(name='nll'),
    custom_acc(),
]

# -----------------------------------------------------------------------------
# Cross-validation scheme
# -----------------------------------------------------------------------------


def get_cv(X, y):
    # use LeaveOneGroupOut cross validation - train on all subjects except for one in each split
    groups = X['subject_id']
    logo = LeaveOneGroupOut()
    return logo.split(X, y,groups)


# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------


def _read_data(path, type_):

    fname = "data_{}.csv".format(type_)
    fp = os.path.join(path, "data/public/", fname)
    data = pd.read_csv(fp)
    data = data.fillna(method='ffill', axis=0) # replace Nan values with previous row value

    fname = "labels_{}.csv".format(type_)
    fp = os.path.join(path, "data/public/", fname)
    labels = pd.read_csv(fp)
    labels = labels.fillna(method='ffill', axis=0)

    return data, labels.to_numpy().ravel()


def get_train_data(path="."):
    return _read_data(path, "train")


def get_test_data(path="."):
    return _read_data(path, "test")
