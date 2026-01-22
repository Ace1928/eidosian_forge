from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_new_bounds_type(self):

    def f(x):
        return x[0] ** 2 + x[1] ** 2
    bounds = Bounds([1, 0], [np.inf, np.inf])
    sol = minimize(f, [0, 0], method='slsqp', bounds=bounds)
    assert_(sol.success)
    assert_allclose(sol.x, [1, 0])