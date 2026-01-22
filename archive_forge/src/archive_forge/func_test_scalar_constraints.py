from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_scalar_constraints(self):
    x = fmin_slsqp(lambda z: z ** 2, [3.0], ieqcons=[lambda z: z[0] - 1], iprint=0)
    assert_array_almost_equal(x, [1.0])
    x = fmin_slsqp(lambda z: z ** 2, [3.0], f_ieqcons=lambda z: [z[0] - 1], iprint=0)
    assert_array_almost_equal(x, [1.0])