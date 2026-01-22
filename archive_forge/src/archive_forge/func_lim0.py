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
def lim0(x1, x2, x3, t0, t1):
    return [scale * (x1 ** 2 + x2 + np.cos(x3) * t0 * t1 + 1) - 1, scale * (x1 ** 2 + x2 + np.cos(x3) * t0 * t1 + 1) + 1]