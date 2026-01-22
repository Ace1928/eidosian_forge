import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_axis_out_of_range(self):
    s = (2, 3, 4, 5)
    a = np.empty(s)
    assert_raises(np.AxisError, expand_dims, a, -6)
    assert_raises(np.AxisError, expand_dims, a, 5)
    a = np.empty((3, 3, 3))
    assert_raises(np.AxisError, expand_dims, a, (0, -6))
    assert_raises(np.AxisError, expand_dims, a, (0, 5))