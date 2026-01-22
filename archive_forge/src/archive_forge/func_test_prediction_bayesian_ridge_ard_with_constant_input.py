from math import log
import numpy as np
import pytest
from sklearn import datasets
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
def test_prediction_bayesian_ridge_ard_with_constant_input():
    n_samples = 4
    n_features = 5
    random_state = check_random_state(42)
    constant_value = random_state.rand()
    X = random_state.random_sample((n_samples, n_features))
    y = np.full(n_samples, constant_value, dtype=np.array(constant_value).dtype)
    expected = np.full(n_samples, constant_value, dtype=np.array(constant_value).dtype)
    for clf in [BayesianRidge(), ARDRegression()]:
        y_pred = clf.fit(X, y).predict(X)
        assert_array_almost_equal(y_pred, expected)