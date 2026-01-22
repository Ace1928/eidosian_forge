import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
def test_broadcasting_shapes(self):
    u = np.ones((2, 1, 3))
    v = np.ones((5, 3))
    assert_equal(np.cross(u, v).shape, (2, 5, 3))
    u = np.ones((10, 3, 5))
    v = np.ones((2, 5))
    assert_equal(np.cross(u, v, axisa=1, axisb=0).shape, (10, 5, 3))
    assert_raises(np.AxisError, np.cross, u, v, axisa=1, axisb=2)
    assert_raises(np.AxisError, np.cross, u, v, axisa=3, axisb=0)
    u = np.ones((10, 3, 5, 7))
    v = np.ones((5, 7, 2))
    assert_equal(np.cross(u, v, axisa=1, axisc=2).shape, (10, 5, 3, 7))
    assert_raises(np.AxisError, np.cross, u, v, axisa=-5, axisb=2)
    assert_raises(np.AxisError, np.cross, u, v, axisa=1, axisb=-4)
    u = np.ones((3, 4, 2))
    for axisc in range(-2, 2):
        assert_equal(np.cross(u, u, axisc=axisc).shape, (3, 4))