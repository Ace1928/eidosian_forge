import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, LIL_CONTAINERS
@pytest.mark.parametrize('with_sample_weight', [True, False])
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_enet_toy_list_input(with_sample_weight, csc_container):
    X = np.array([[-1], [0], [1]])
    X = csc_container(X)
    Y = [-1, 0, 1]
    T = np.array([[2], [3], [4]])
    if with_sample_weight:
        sw = np.array([2.0, 2, 2])
    else:
        sw = None
    clf = ElasticNet(alpha=0, l1_ratio=1.0)
    ignore_warnings(clf.fit)(X, Y, sample_weight=sw)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])
    assert_almost_equal(clf.dual_gap_, 0)
    clf = ElasticNet(alpha=0.5, l1_ratio=0.3)
    clf.fit(X, Y, sample_weight=sw)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0)
    clf = ElasticNet(alpha=0.5, l1_ratio=0.5)
    clf.fit(X, Y, sample_weight=sw)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.45454], 3)
    assert_array_almost_equal(pred, [0.909, 1.3636, 1.8181], 3)
    assert_almost_equal(clf.dual_gap_, 0)