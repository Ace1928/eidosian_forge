import re
import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble._weight_boosting import _samme_proba
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import shuffle
from sklearn.utils._mocking import NoSampleWeightWrapper
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def test_sample_weight_adaboost_regressor():
    """
    AdaBoostRegressor should work without sample_weights in the base estimator
    The random weighted sampling is done internally in the _boost method in
    AdaBoostRegressor.
    """

    class DummyEstimator(BaseEstimator):

        def fit(self, X, y):
            pass

        def predict(self, X):
            return np.zeros(X.shape[0])
    boost = AdaBoostRegressor(DummyEstimator(), n_estimators=3)
    boost.fit(X, y_regr)
    assert len(boost.estimator_weights_) == len(boost.estimator_errors_)