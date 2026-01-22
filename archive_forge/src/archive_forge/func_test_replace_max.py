import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_replace_max(self):
    a_base = np.array([[10, 30, 20], [60, 40, 50]])
    for axis in list(range(a_base.ndim)) + [None]:
        a = a_base.copy()
        i_max = _add_keepdims(np.argmax)(a, axis=axis)
        put_along_axis(a, i_max, -99, axis=axis)
        i_min = _add_keepdims(np.argmin)(a, axis=axis)
        assert_equal(i_min, i_max)