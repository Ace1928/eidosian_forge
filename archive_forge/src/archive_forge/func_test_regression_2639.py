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
def test_regression_2639(self):
    x = [574.142, 574.154, 574.165, 574.177, 574.188, 574.199, 574.211, 574.222, 574.234, 574.245]
    y = [859.0, 997.0, 1699.0, 2604.0, 2013.0, 1964.0, 2435.0, 1550.0, 949.0, 841.0]
    guess = [574.1861428571428, 574.2155714285715, 1302.0, 1302.0, 0.0035019999999983615, 859.0]
    good = [574.17715, 574.209188, 1741.87044, 1586.46166, 0.010068462, 857.450661]

    def f_double_gauss(x, x0, x1, A0, A1, sigma, c):
        return A0 * np.exp(-(x - x0) ** 2 / (2.0 * sigma ** 2)) + A1 * np.exp(-(x - x1) ** 2 / (2.0 * sigma ** 2)) + c
    popt, pcov = curve_fit(f_double_gauss, x, y, guess, maxfev=10000)
    assert_allclose(popt, good, rtol=1e-05)