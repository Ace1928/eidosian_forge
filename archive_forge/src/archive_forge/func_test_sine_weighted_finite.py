import sys
import math
import numpy as np
from numpy import sqrt, cos, sin, arctan, exp, log, pi
from numpy.testing import (assert_,
import pytest
from scipy.integrate import quad, dblquad, tplquad, nquad
from scipy.special import erf, erfc
from scipy._lib._ccallback import LowLevelCallable
import ctypes
import ctypes.util
from scipy._lib._ccallback_c import sine_ctypes
import scipy.integrate._test_multivariate as clib_test
def test_sine_weighted_finite(self):

    def myfunc(x, a):
        return exp(a * (x - 1))
    ome = 2.0 ** 3.4
    assert_quad(quad(myfunc, 0, 1, args=20, weight='sin', wvar=ome), (20 * sin(ome) - ome * cos(ome) + ome * exp(-20)) / (20 ** 2 + ome ** 2))