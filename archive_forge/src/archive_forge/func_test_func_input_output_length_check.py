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
def test_func_input_output_length_check(self):

    def func(x):
        return 2 * (x[0] - 3) ** 2 + 1
    with assert_raises(TypeError, match='Improper input: func input vector length N='):
        optimize.leastsq(func, x0=[0, 1])