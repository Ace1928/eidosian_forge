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
def test_singular(self):

    def myfunc(x):
        if 0 < x < 2.5:
            return sin(x)
        elif 2.5 <= x <= 5.0:
            return exp(-x)
        else:
            return 0.0
    assert_quad(quad(myfunc, 0, 10, points=[2.5, 5.0]), 1 - cos(2.5) + exp(-2.5) - exp(-5.0))