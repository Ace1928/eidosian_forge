import numpy as np
from numpy.testing import (assert_, assert_array_almost_equal,
from numpy import linspace, sin, cos, random, exp, allclose
from scipy.interpolate._rbf import Rbf
def test_2drbf_regularity():
    tolerances = {'multiquadric': 0.1, 'inverse multiquadric': 0.15, 'gaussian': 0.15, 'cubic': 0.15, 'quintic': 0.1, 'thin-plate': 0.15, 'linear': 0.2}
    for function in FUNCTIONS:
        check_2drbf1d_regularity(function, tolerances.get(function, 0.01))