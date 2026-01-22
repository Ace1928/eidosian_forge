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
@pytest.mark.parametrize('rs_interface', [True, False])
def test_function_calls(solver_name, rs_interface):
    solver = (lambda f, a, b, **kwargs: root_scalar(f, bracket=(a, b))) if rs_interface else getattr(zeros, solver_name)

    def f(x):
        f.calls += 1
        return x ** 2 - 1
    f.calls = 0
    res = solver(f, 0, 10, full_output=True)
    if rs_interface:
        assert res.function_calls == f.calls
    else:
        assert res[1].function_calls == f.calls