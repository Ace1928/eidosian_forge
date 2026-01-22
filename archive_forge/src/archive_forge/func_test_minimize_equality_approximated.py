from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_minimize_equality_approximated(self):
    jacs = [None, False, '2-point', '3-point']
    for jac in jacs:
        res = minimize(self.fun, [-1.0, 1.0], args=(-1.0,), jac=jac, constraints={'type': 'eq', 'fun': self.f_eqcon, 'args': (-1.0,)}, method='SLSQP', options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [1, 1])