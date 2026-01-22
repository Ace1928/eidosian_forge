from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_inconsistent_inequalities(self):

    def cost(x):
        return -1 * x[0] + 4 * x[1]

    def ineqcons1(x):
        return x[1] - x[0] - 1

    def ineqcons2(x):
        return x[0] - x[1]
    x0 = (1, 5)
    bounds = ((-5, 5), (-5, 5))
    cons = (dict(type='ineq', fun=ineqcons1), dict(type='ineq', fun=ineqcons2))
    res = minimize(cost, x0, method='SLSQP', bounds=bounds, constraints=cons)
    assert_(not res.success)