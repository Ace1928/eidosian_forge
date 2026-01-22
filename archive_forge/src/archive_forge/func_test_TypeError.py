import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def test_TypeError(self):
    i = 1
    assert_raises(TypeError, basinhopping, func2d, self.x0[i], take_step=1)
    assert_raises(TypeError, basinhopping, func2d, self.x0[i], accept_test=1)