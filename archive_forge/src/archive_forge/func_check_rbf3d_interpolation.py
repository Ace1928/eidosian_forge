import numpy as np
from numpy.testing import (assert_, assert_array_almost_equal,
from numpy import linspace, sin, cos, random, exp, allclose
from scipy.interpolate._rbf import Rbf
def check_rbf3d_interpolation(function):
    x = random.rand(50, 1) * 4 - 2
    y = random.rand(50, 1) * 4 - 2
    z = random.rand(50, 1) * 4 - 2
    d = x * exp(-x ** 2 - y ** 2)
    rbf = Rbf(x, y, z, d, epsilon=2, function=function)
    di = rbf(x, y, z)
    di.shape = x.shape
    assert_array_almost_equal(di, d)