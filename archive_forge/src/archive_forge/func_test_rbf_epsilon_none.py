import numpy as np
from numpy.testing import (assert_, assert_array_almost_equal,
from numpy import linspace, sin, cos, random, exp, allclose
from scipy.interpolate._rbf import Rbf
def test_rbf_epsilon_none():
    x = linspace(0, 10, 9)
    y = sin(x)
    Rbf(x, y, epsilon=None)