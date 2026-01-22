import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_stratified_strategy(global_random_seed):
    X = [[0]] * 5
    y = [1, 2, 1, 1, 2]
    clf = DummyClassifier(strategy='stratified', random_state=global_random_seed)
    clf.fit(X, y)
    X = [[0]] * 500
    y_pred = clf.predict(X)
    p = np.bincount(y_pred) / float(len(X))
    assert_almost_equal(p[1], 3.0 / 5, decimal=1)
    assert_almost_equal(p[2], 2.0 / 5, decimal=1)
    _check_predict_proba(clf, X, y)