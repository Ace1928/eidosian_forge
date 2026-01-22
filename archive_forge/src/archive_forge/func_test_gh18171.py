import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
@pytest.mark.parametrize('method', ['brentq', 'brenth', 'bisect', 'ridder', 'toms748'])
def test_gh18171(method):

    def f(x):
        f._count += 1
        return np.nan
    f._count = 0
    res = root_scalar(f, bracket=(0, 1), method=method)
    assert res.converged is False
    assert res.flag.startswith('The function value at x')
    assert res.function_calls == f._count
    assert str(res.root) in res.flag