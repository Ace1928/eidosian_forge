import numpy as np
from numpy.testing import (assert_, assert_array_almost_equal,
from numpy import linspace, sin, cos, random, exp, allclose
from scipy.interpolate._rbf import Rbf
def test_function_is_callable():
    x = linspace(0, 10, 9)
    y = sin(x)

    def linfunc(x):
        return x
    rbf = Rbf(x, y, function=linfunc)
    yi = rbf(x)
    assert_array_almost_equal(y, yi)