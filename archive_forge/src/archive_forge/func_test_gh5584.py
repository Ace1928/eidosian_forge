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
def test_gh5584(solver_name, rs_interface):
    solver = (lambda f, a, b, **kwargs: root_scalar(f, bracket=(a, b))) if rs_interface else getattr(zeros, solver_name)

    def f(x):
        return 1e-200 * x
    with pytest.raises(ValueError, match='...must have different signs'):
        solver(f, -0.5, -0.4, full_output=True)
    res = solver(f, -0.5, 0.4, full_output=True)
    res = res if rs_interface else res[1]
    assert res.converged
    assert_allclose(res.root, 0, atol=1e-08)
    res = solver(f, -0.5, float('-0.0'), full_output=True)
    res = res if rs_interface else res[1]
    assert res.converged
    assert_allclose(res.root, 0, atol=1e-08)