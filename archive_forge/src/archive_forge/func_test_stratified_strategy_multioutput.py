import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_stratified_strategy_multioutput(global_random_seed):
    X = [[0]] * 5
    y = np.array([[2, 1], [2, 2], [1, 1], [1, 2], [1, 1]])
    clf = DummyClassifier(strategy='stratified', random_state=global_random_seed)
    clf.fit(X, y)
    X = [[0]] * 500
    y_pred = clf.predict(X)
    for k in range(y.shape[1]):
        p = np.bincount(y_pred[:, k]) / float(len(X))
        assert_almost_equal(p[1], 3.0 / 5, decimal=1)
        assert_almost_equal(p[2], 2.0 / 5, decimal=1)
        _check_predict_proba(clf, X, y)
    _check_behavior_2d(clf)