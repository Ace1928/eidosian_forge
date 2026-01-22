from numpy.testing import assert_, assert_equal
import pytest
from pytest import raises as assert_raises, warns as assert_warns
import numpy as np
from scipy.optimize import root
def test_tol_parameter(self):

    def func(z):
        x, y = z
        return np.array([x ** 3 - 1, y ** 3 - 1])

    def dfunc(z):
        x, y = z
        return np.array([[3 * x ** 2, 0], [0, 3 * y ** 2]])
    for method in ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson', 'diagbroyden', 'krylov']:
        if method in ('linearmixing', 'excitingmixing'):
            continue
        if method in ('hybr', 'lm'):
            jac = dfunc
        else:
            jac = None
        sol1 = root(func, [1.1, 1.1], jac=jac, tol=0.0001, method=method)
        sol2 = root(func, [1.1, 1.1], jac=jac, tol=0.5, method=method)
        msg = f'{method}: {func(sol1.x)} vs. {func(sol2.x)}'
        assert_(sol1.success, msg)
        assert_(sol2.success, msg)
        assert_(abs(func(sol1.x)).max() < abs(func(sol2.x)).max(), msg)