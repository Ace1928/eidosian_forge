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
def test_ransac_final_model_fit_sample_weight():
    X, y = make_regression(n_samples=1000, random_state=10)
    rng = check_random_state(42)
    sample_weight = rng.randint(1, 4, size=y.shape[0])
    sample_weight = sample_weight / sample_weight.sum()
    ransac = RANSACRegressor(estimator=LinearRegression(), random_state=0)
    ransac.fit(X, y, sample_weight=sample_weight)
    final_model = LinearRegression()
    mask_samples = ransac.inlier_mask_
    final_model.fit(X[mask_samples], y[mask_samples], sample_weight=sample_weight[mask_samples])
    assert_allclose(ransac.estimator_.coef_, final_model.coef_, atol=1e-12)