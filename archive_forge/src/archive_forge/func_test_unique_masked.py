import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_unique_masked(self):
    x = np.array([64, 0, 1, 2, 3, 63, 63, 0, 0, 0, 1, 2, 0, 63, 0], dtype='uint8')
    y = np.ma.masked_equal(x, 0)
    v = np.unique(y)
    v2, i, c = np.unique(y, return_index=True, return_counts=True)
    msg = 'Unique returned different results when asked for index'
    assert_array_equal(v.data, v2.data, msg)
    assert_array_equal(v.mask, v2.mask, msg)