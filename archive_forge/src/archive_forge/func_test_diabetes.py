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
@pytest.mark.parametrize('loss', ['linear', 'square', 'exponential'])
def test_diabetes(loss):
    reg = AdaBoostRegressor(loss=loss, random_state=0)
    reg.fit(diabetes.data, diabetes.target)
    score = reg.score(diabetes.data, diabetes.target)
    assert score > 0.55
    assert len(reg.estimators_) > 1
    assert len(set((est.random_state for est in reg.estimators_))) == len(reg.estimators_)