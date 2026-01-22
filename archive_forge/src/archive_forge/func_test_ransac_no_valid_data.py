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
def test_ransac_no_valid_data():

    def is_data_valid(X, y):
        return False
    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(estimator, is_data_valid=is_data_valid, max_trials=5)
    msg = 'RANSAC could not find a valid consensus set'
    with pytest.raises(ValueError, match=msg):
        ransac_estimator.fit(X, y)
    assert ransac_estimator.n_skips_no_inliers_ == 0
    assert ransac_estimator.n_skips_invalid_data_ == 5
    assert ransac_estimator.n_skips_invalid_model_ == 0