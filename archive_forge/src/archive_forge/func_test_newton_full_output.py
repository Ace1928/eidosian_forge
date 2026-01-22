import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_newton_full_output(self):
    x0 = 3
    expected_counts = [(6, 7), (5, 10), (3, 9)]
    for derivs in range(3):
        kwargs = {'tol': 1e-06, 'full_output': True}
        for k, v in [['fprime', f1_1], ['fprime2', f1_2]][:derivs]:
            kwargs[k] = v
        x, r = zeros.newton(f1, x0, disp=False, **kwargs)
        assert_(r.converged)
        assert_equal(x, r.root)
        assert_equal((r.iterations, r.function_calls), expected_counts[derivs])
        if derivs == 0:
            assert r.function_calls <= r.iterations + 1
        else:
            assert_equal(r.function_calls, (derivs + 1) * r.iterations)
        iters = r.iterations - 1
        x, r = zeros.newton(f1, x0, maxiter=iters, disp=False, **kwargs)
        assert_(not r.converged)
        assert_equal(x, r.root)
        assert_equal(r.iterations, iters)
        if derivs == 1:
            msg = 'Failed to converge after %d iterations, value is .*' % iters
            with pytest.raises(RuntimeError, match=msg):
                x, r = zeros.newton(f1, x0, maxiter=iters, disp=True, **kwargs)