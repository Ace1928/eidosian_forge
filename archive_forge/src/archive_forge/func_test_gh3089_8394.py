import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
@pytest.mark.parametrize('solver_name', ['brentq', 'brenth', 'bisect', 'ridder', 'toms748'])
def test_gh3089_8394(solver_name):

    def f(x):
        return np.nan
    solver = getattr(zeros, solver_name)
    with pytest.raises(ValueError, match='The function value at x...'):
        solver(f, 0, 1)