import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_stratified_strategy_sparse_target(global_random_seed, csc_container):
    X = [[0]] * 5
    y = csc_container(np.array([[4, 1], [0, 0], [1, 1], [1, 4], [1, 1]]))
    clf = DummyClassifier(strategy='stratified', random_state=global_random_seed)
    clf.fit(X, y)
    X = [[0]] * 500
    y_pred = clf.predict(X)
    assert sp.issparse(y_pred)
    y_pred = y_pred.toarray()
    for k in range(y.shape[1]):
        p = np.bincount(y_pred[:, k]) / float(len(X))
        assert_almost_equal(p[1], 3.0 / 5, decimal=1)
        assert_almost_equal(p[0], 1.0 / 5, decimal=1)
        assert_almost_equal(p[4], 1.0 / 5, decimal=1)