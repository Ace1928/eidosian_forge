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
def test_b_less_than_a_full_output(self):

    def f(x):
        return 1.0
    res_1 = quad(f, 0, 1, weight='alg', wvar=(0, 0), full_output=True)
    res_2 = quad(f, 1, 0, weight='alg', wvar=(0, 0), full_output=True)
    err = max(res_1[1], res_2[1])
    assert_allclose(res_1[0], -res_2[0], atol=err)