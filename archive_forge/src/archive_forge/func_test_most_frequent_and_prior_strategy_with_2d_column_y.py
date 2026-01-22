import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_most_frequent_and_prior_strategy_with_2d_column_y():
    X = [[0], [0], [0], [0]]
    y_1d = [1, 2, 1, 1]
    y_2d = [[1], [2], [1], [1]]
    for strategy in ('most_frequent', 'prior'):
        clf_1d = DummyClassifier(strategy=strategy, random_state=0)
        clf_2d = DummyClassifier(strategy=strategy, random_state=0)
        clf_1d.fit(X, y_1d)
        clf_2d.fit(X, y_2d)
        assert_array_equal(clf_1d.predict(X), clf_2d.predict(X))