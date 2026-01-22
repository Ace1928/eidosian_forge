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
def assert_quad(value_and_err, tabled_value, error_tolerance=1.5e-08):
    value, err = value_and_err
    assert_allclose(value, tabled_value, atol=err, rtol=0)
    if error_tolerance is not None:
        assert_array_less(err, error_tolerance)