import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, LIL_CONTAINERS
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_enet_multitarget(csc_container):
    n_targets = 3
    X, y = make_sparse_data(csc_container, n_targets=n_targets)
    estimator = ElasticNet(alpha=0.01, precompute=False)
    estimator.fit(X, y)
    coef, intercept, dual_gap = (estimator.coef_, estimator.intercept_, estimator.dual_gap_)
    for k in range(n_targets):
        estimator.fit(X, y[:, k])
        assert_array_almost_equal(coef[k, :], estimator.coef_)
        assert_array_almost_equal(intercept[k], estimator.intercept_)
        assert_array_almost_equal(dual_gap[k], estimator.dual_gap_)