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
def test_no_acceleration(self):
    ks = 2
    kl = 6
    m = 1.3
    n0 = 1.001
    i0 = (m - 1) / m * (kl / ks / m) ** (1 / (m - 1))

    def func(n):
        return np.log(kl / ks / n) / np.log(i0 * n / (n - 1)) + 1
    n = fixed_point(func, n0, method='iteration')
    assert_allclose(n, m)