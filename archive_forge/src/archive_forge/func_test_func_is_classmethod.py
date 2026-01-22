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
def test_func_is_classmethod(self):

    class test_self:
        """This class tests if curve_fit passes the correct number of
               arguments when the model function is a class instance method.
            """

        def func(self, x, a, b):
            return b * x ** a
    test_self_inst = test_self()
    popt, pcov = curve_fit(test_self_inst.func, self.x, self.y)
    assert_(pcov.shape == (2, 2))
    assert_array_almost_equal(popt, [1.7989, 1.1642], decimal=4)
    assert_array_almost_equal(pcov, [[0.0852, -0.126], [-0.126, 0.1912]], decimal=4)