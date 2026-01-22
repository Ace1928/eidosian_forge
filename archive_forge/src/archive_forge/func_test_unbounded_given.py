from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_unbounded_given(self):
    res = fmin_slsqp(self.fun, [-1.0, 1.0], args=(-1.0,), fprime=self.jac, iprint=0, full_output=1)
    x, fx, its, imode, smode = res
    assert_(imode == 0, imode)
    assert_array_almost_equal(x, [2, 1])