import logging
import sys
import numpy
import numpy as np
import time
from multiprocessing import Pool
from numpy.testing import assert_allclose, IS_PYPY
import pytest
from pytest import raises as assert_raises, warns
from scipy.optimize import (shgo, Bounds, minimize_scalar, minimize, rosen,
from scipy.optimize._constraints import new_constraint_to_old
from scipy.optimize._shgo import SHGO
def test_21_1_jac_true(self):
    """Test that shgo can handle objective functions that return the
        gradient alongside the objective value. Fixes gh-13547"""

    def func(x):
        return (numpy.sum(numpy.power(x, 2)), 2 * x)
    shgo(func, bounds=[[-1, 1], [1, 2]], n=100, iters=5, sampling_method='sobol', minimizer_kwargs={'method': 'SLSQP', 'jac': True})

    def func(x):
        return (numpy.sum(x ** 2), 2 * x)
    bounds = [[-1, 1], [1, 2], [-1, 1], [1, 2], [0, 3]]
    res = shgo(func, bounds=bounds, sampling_method='sobol', minimizer_kwargs={'method': 'SLSQP', 'jac': True})
    ref = minimize(func, x0=[1, 1, 1, 1, 1], bounds=bounds, jac=True)
    assert res.success
    assert_allclose(res.fun, ref.fun)
    assert_allclose(res.x, ref.x, atol=1e-15)