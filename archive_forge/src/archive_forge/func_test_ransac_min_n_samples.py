import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._ransac import _dynamic_max_trials
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
def test_ransac_min_n_samples():
    estimator = LinearRegression()
    ransac_estimator1 = RANSACRegressor(estimator, min_samples=2, residual_threshold=5, random_state=0)
    ransac_estimator2 = RANSACRegressor(estimator, min_samples=2.0 / X.shape[0], residual_threshold=5, random_state=0)
    ransac_estimator5 = RANSACRegressor(estimator, min_samples=2, residual_threshold=5, random_state=0)
    ransac_estimator6 = RANSACRegressor(estimator, residual_threshold=5, random_state=0)
    ransac_estimator7 = RANSACRegressor(estimator, min_samples=X.shape[0] + 1, residual_threshold=5, random_state=0)
    ransac_estimator8 = RANSACRegressor(Ridge(), min_samples=None, residual_threshold=5, random_state=0)
    ransac_estimator1.fit(X, y)
    ransac_estimator2.fit(X, y)
    ransac_estimator5.fit(X, y)
    ransac_estimator6.fit(X, y)
    assert_array_almost_equal(ransac_estimator1.predict(X), ransac_estimator2.predict(X))
    assert_array_almost_equal(ransac_estimator1.predict(X), ransac_estimator5.predict(X))
    assert_array_almost_equal(ransac_estimator1.predict(X), ransac_estimator6.predict(X))
    with pytest.raises(ValueError):
        ransac_estimator7.fit(X, y)
    err_msg = '`min_samples` needs to be explicitly set'
    with pytest.raises(ValueError, match=err_msg):
        ransac_estimator8.fit(X, y)