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
def test_triple_integral(self):

    def simpfunc(z, y, x, t):
        return (x + y + z) * t
    a, b = (1.0, 2.0)
    assert_quad(tplquad(simpfunc, a, b, lambda x: x, lambda x: 2 * x, lambda x, y: x - y, lambda x, y: x + y, (2.0,)), 2 * 8 / 3.0 * (b ** 4.0 - a ** 4.0))