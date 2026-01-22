import numpy as np
from numpy.testing import (assert_, assert_array_almost_equal,
from numpy import linspace, sin, cos, random, exp, allclose
from scipy.interpolate._rbf import Rbf
def test_2drbf_interpolation():
    for function in FUNCTIONS:
        check_2drbf1d_interpolation(function)
        check_2drbf2d_interpolation(function)
        check_2drbf3d_interpolation(function)