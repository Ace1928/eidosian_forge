import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_newton_does_not_modify_x0(self):
    x0 = np.array([0.1, 3])
    x0_copy = x0.copy()
    newton(np.sin, x0, np.cos)
    assert_array_equal(x0, x0_copy)