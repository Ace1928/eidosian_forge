from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_minimize_unbounded_given(self):
    res = minimize(self.fun, [-1.0, 1.0], args=(-1.0,), jac=self.jac, method='SLSQP', options=self.opts)
    assert_(res['success'], res['message'])
    assert_allclose(res.x, [2, 1])