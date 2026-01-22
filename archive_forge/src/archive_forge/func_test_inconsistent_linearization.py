from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_inconsistent_linearization(self):
    x = [0, 1]

    def f1(x):
        return x[0] + x[1] - 2

    def f2(x):
        return x[0] ** 2 - 1
    sol = minimize(lambda x: x[0] ** 2 + x[1] ** 2, x, constraints=({'type': 'eq', 'fun': f1}, {'type': 'ineq', 'fun': f2}), bounds=((0, None), (0, None)), method='SLSQP')
    x = sol.x
    assert_allclose(f1(x), 0, atol=1e-08)
    assert_(f2(x) >= -1e-08)
    assert_(sol.success, sol)