import os
import re
import sys
from contextlib import contextmanager
import numpy as np
import pytest
from numpy.testing import (
from scipy.linalg import norm
from scipy.optimize import fmin_bfgs
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model._theil_sen import (
from sklearn.utils._testing import assert_almost_equal
def test_spatial_median_2d():
    X = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0]).reshape(3, 2)
    _, median = _spatial_median(X, max_iter=100, tol=1e-06)

    def cost_func(y):
        dists = np.array([norm(x - y) for x in X])
        return np.sum(dists)
    fermat_weber = fmin_bfgs(cost_func, median, disp=False)
    assert_array_almost_equal(median, fermat_weber)
    warning_message = 'Maximum number of iterations 30 reached in spatial median.'
    with pytest.warns(ConvergenceWarning, match=warning_message):
        _spatial_median(X, max_iter=30, tol=0.0)