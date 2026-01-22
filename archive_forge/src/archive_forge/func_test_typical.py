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
def test_typical(self):

    def myfunc(x, n, z):
        return cos(n * x - z * sin(x)) / pi
    assert_quad(quad(myfunc, 0, pi, (2, 1.8)), 0.30614353532540295)