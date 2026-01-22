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
def test_square_aliased_ranges_and_opts(self):

    def f(y, x):
        return 1.0
    r = [-1, 1]
    opt = {}
    assert_quad(nquad(f, [r, r], opts=[opt, opt]), 4.0)