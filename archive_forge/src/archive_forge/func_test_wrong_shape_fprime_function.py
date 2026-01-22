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
def test_wrong_shape_fprime_function(self):

    def func(x):
        return dummy_func(x, (2,))

    def deriv_func(x):
        return dummy_func(x, (3, 3))
    assert_raises(TypeError, optimize.fsolve, func, x0=[0, 1], fprime=deriv_func)