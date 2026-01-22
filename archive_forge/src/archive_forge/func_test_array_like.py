import warnings
import pytest
from numpy.testing import (assert_, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
import numpy as np
from numpy import array, float64
from multiprocessing.pool import ThreadPool
from scipy import optimize, linalg
from scipy.special import lambertw
from scipy.optimize._minpack_py import leastsq, curve_fit, fixed_point
from scipy.optimize import OptimizeWarning
from scipy.optimize._minimize import Bounds
def test_array_like(self):

    def f_linear(x, a, b):
        return a * x + b
    x = [1, 2, 3, 4]
    y = [3, 5, 7, 9]
    assert_allclose(curve_fit(f_linear, x, y)[0], [2, 1], atol=1e-10)