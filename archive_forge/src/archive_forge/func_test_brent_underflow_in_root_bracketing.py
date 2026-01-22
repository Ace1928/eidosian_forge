import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_brent_underflow_in_root_bracketing():
    underflow_scenario = (-450.0, -350.0, -400.0)
    overflow_scenario = (350.0, 450.0, 400.0)
    for a, b, root in [underflow_scenario, overflow_scenario]:
        c = np.exp(root)
        for method in [zeros.brenth, zeros.brentq]:
            res = method(lambda x: np.exp(x) - c, a, b)
            assert_allclose(root, res)