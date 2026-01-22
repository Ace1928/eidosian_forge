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
def test_NaN_handling(self):
    xdata = np.array([1, np.nan, 3])
    ydata = np.array([1, 2, 3])
    assert_raises(ValueError, curve_fit, lambda x, a, b: a * x + b, xdata, ydata)
    assert_raises(ValueError, curve_fit, lambda x, a, b: a * x + b, ydata, xdata)
    assert_raises(ValueError, curve_fit, lambda x, a, b: a * x + b, xdata, ydata, **{'check_finite': True})