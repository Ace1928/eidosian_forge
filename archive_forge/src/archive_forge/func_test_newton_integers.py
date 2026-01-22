import numpy as np
from numpy.testing import assert_almost_equal
from pytest import raises as assert_raises
import scipy.optimize
def test_newton_integers(self):
    root = scipy.optimize.newton(lambda x: x ** 2 - 1, x0=2, fprime=lambda x: 2 * x)
    assert_almost_equal(root, 1.0)