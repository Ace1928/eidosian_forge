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
@pytest.mark.parametrize('alpha, fit_intercept, positive', [(0.1, False, False), (0.1, True, False), (0.001, False, True), (0.001, True, True)])
def test_sparse_enet_not_as_toy_dataset(csc_container, alpha, fit_intercept, positive):
    n_samples, n_features, max_iter = (100, 100, 1000)
    n_informative = 10
    X, y = make_sparse_data(csc_container, n_samples, n_features, n_informative, positive=positive)
    X_train, X_test = (X[n_samples // 2:], X[:n_samples // 2])
    y_train, y_test = (y[n_samples // 2:], y[:n_samples // 2])
    s_clf = ElasticNet(alpha=alpha, l1_ratio=0.8, fit_intercept=fit_intercept, max_iter=max_iter, tol=1e-07, positive=positive, warm_start=True)
    s_clf.fit(X_train, y_train)
    assert_almost_equal(s_clf.dual_gap_, 0, 4)
    assert s_clf.score(X_test, y_test) > 0.85
    d_clf = ElasticNet(alpha=alpha, l1_ratio=0.8, fit_intercept=fit_intercept, max_iter=max_iter, tol=1e-07, positive=positive, warm_start=True)
    d_clf.fit(X_train.toarray(), y_train)
    assert_almost_equal(d_clf.dual_gap_, 0, 4)
    assert d_clf.score(X_test, y_test) > 0.85
    assert_almost_equal(s_clf.coef_, d_clf.coef_, 5)
    assert_almost_equal(s_clf.intercept_, d_clf.intercept_, 5)
    assert np.sum(s_clf.coef_ != 0.0) < 2 * n_informative