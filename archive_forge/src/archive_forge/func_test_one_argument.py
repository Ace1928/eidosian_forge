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
def test_one_argument(self):

    def func(x, a):
        return x ** a
    popt, pcov = curve_fit(func, self.x, self.y)
    assert_(len(popt) == 1)
    assert_(pcov.shape == (1, 1))
    assert_almost_equal(popt[0], 1.9149, decimal=4)
    assert_almost_equal(pcov[0, 0], 0.0016, decimal=4)
    res = curve_fit(func, self.x, self.y, full_output=1, check_finite=False)
    popt2, pcov2, infodict, errmsg, ier = res
    assert_array_almost_equal(popt, popt2)