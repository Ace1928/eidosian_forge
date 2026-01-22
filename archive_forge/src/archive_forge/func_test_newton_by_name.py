import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_newton_by_name(self):
    """Invoke newton through root_scalar()"""
    for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
        r = root_scalar(f, method='newton', x0=3, fprime=f_1, xtol=1e-06)
        assert_allclose(f(r.root), 0, atol=1e-06)
    for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
        r = root_scalar(f, method='newton', x0=3, xtol=1e-06)
        assert_allclose(f(r.root), 0, atol=1e-06)