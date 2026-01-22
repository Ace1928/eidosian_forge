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
def test_float32(self):

    def func(p, x, y):
        q = p[0] * np.exp(-(x - p[1]) ** 2 / (2.0 * p[2] ** 2)) + p[3]
        return q - y
    x = np.array([1.475, 1.429, 1.409, 1.419, 1.455, 1.519, 1.472, 1.368, 1.286, 1.231], dtype=np.float32)
    y = np.array([0.0168, 0.0193, 0.0211, 0.0202, 0.0171, 0.0151, 0.0185, 0.0258, 0.034, 0.0396], dtype=np.float32)
    p0 = np.array([1.0, 1.0, 1.0, 1.0])
    p1, success = optimize.leastsq(func, p0, args=(x, y))
    assert_(success in [1, 2, 3, 4])
    assert_((func(p1, x, y) ** 2).sum() < 0.0001 * (func(p0, x, y) ** 2).sum())