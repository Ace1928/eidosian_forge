from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_minimize_bound_equality_given2(self):
    res = minimize(self.fun, [-1.0, 1.0], method='SLSQP', jac=self.jac, args=(-1.0,), bounds=[(-0.8, 1.0), (-1, 0.8)], constraints={'type': 'eq', 'fun': self.f_eqcon, 'args': (-1.0,), 'jac': self.fprime_eqcon}, options=self.opts)
    assert_(res['success'], res['message'])
    assert_allclose(res.x, [0.8, 0.8], atol=0.001)
    assert_(-0.8 <= res.x[0] <= 1)
    assert_(-1 <= res.x[1] <= 0.8)