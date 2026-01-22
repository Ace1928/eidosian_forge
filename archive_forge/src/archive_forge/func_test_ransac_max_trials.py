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
def test_ransac_max_trials():
    estimator = LinearRegression()
    ransac_estimator = RANSACRegressor(estimator, min_samples=2, residual_threshold=5, max_trials=0, random_state=0)
    with pytest.raises(ValueError):
        ransac_estimator.fit(X, y)
    max_trials = _dynamic_max_trials(len(X) - len(outliers), X.shape[0], 2, 1 - 1e-09)
    ransac_estimator = RANSACRegressor(estimator, min_samples=2)
    for i in range(50):
        ransac_estimator.set_params(min_samples=2, random_state=i)
        ransac_estimator.fit(X, y)
        assert ransac_estimator.n_trials_ < max_trials + 1