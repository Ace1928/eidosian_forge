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
def test_dtypes2(self):

    def hyperbola(x, s_1, s_2, o_x, o_y, c):
        b_2 = (s_1 + s_2) / 2
        b_1 = (s_2 - s_1) / 2
        return o_y + b_1 * (x - o_x) + b_2 * np.sqrt((x - o_x) ** 2 + c ** 2 / 4)
    min_fit = np.array([-3.0, 0.0, -2.0, -10.0, 0.0])
    max_fit = np.array([0.0, 3.0, 3.0, 0.0, 10.0])
    guess = np.array([-2.5 / 3.0, 4 / 3.0, 1.0, -4.0, 0.5])
    params = [-2, 0.4, -1, -5, 9.5]
    xdata = np.array([-32, -16, -8, 4, 4, 8, 16, 32])
    ydata = hyperbola(xdata, *params)
    popt_64, _ = curve_fit(f=hyperbola, xdata=xdata, ydata=ydata, p0=guess, bounds=(min_fit, max_fit))
    xdata = xdata.astype(np.float32)
    ydata = hyperbola(xdata, *params)
    popt_32, _ = curve_fit(f=hyperbola, xdata=xdata, ydata=ydata, p0=guess, bounds=(min_fit, max_fit))
    assert_allclose(popt_32, popt_64, atol=2e-05)