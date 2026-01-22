import numpy as np
from numpy.testing import (assert_, assert_array_almost_equal,
from numpy import linspace, sin, cos, random, exp, allclose
from scipy.interpolate._rbf import Rbf
def test_rbf_stability():
    for function in FUNCTIONS:
        check_rbf1d_stability(function)