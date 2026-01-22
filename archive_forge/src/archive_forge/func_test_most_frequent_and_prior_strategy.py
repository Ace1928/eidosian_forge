import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_most_frequent_and_prior_strategy():
    X = [[0], [0], [0], [0]]
    y = [1, 2, 1, 1]
    for strategy in ('most_frequent', 'prior'):
        clf = DummyClassifier(strategy=strategy, random_state=0)
        clf.fit(X, y)
        assert_array_equal(clf.predict(X), np.ones(len(X)))
        _check_predict_proba(clf, X, y)
        if strategy == 'prior':
            assert_array_almost_equal(clf.predict_proba([X[0]]), clf.class_prior_.reshape((1, -1)))
        else:
            assert_array_almost_equal(clf.predict_proba([X[0]]), clf.class_prior_.reshape((1, -1)) > 0.5)