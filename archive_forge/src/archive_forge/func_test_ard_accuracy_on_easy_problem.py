from math import log
import numpy as np
import pytest
from sklearn import datasets
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
@pytest.mark.parametrize('n_samples, n_features', ((10, 100), (100, 10)))
def test_ard_accuracy_on_easy_problem(global_random_seed, n_samples, n_features):
    X = np.random.RandomState(global_random_seed).normal(size=(250, 3))
    y = X[:, 1]
    regressor = ARDRegression()
    regressor.fit(X, y)
    abs_coef_error = np.abs(1 - regressor.coef_[1])
    assert abs_coef_error < 1e-10