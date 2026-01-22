import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_axis_insertion_ma(self):

    def f1to2(x):
        """produces an asymmetric non-square matrix from x"""
        assert_equal(x.ndim, 1)
        res = x[::-1] * x[1:, None]
        return np.ma.masked_where(res % 5 == 0, res)
    a = np.arange(6 * 3).reshape((6, 3))
    res = apply_along_axis(f1to2, 0, a)
    assert_(isinstance(res, np.ma.masked_array))
    assert_equal(res.ndim, 3)
    assert_array_equal(res[:, :, 0].mask, f1to2(a[:, 0]).mask)
    assert_array_equal(res[:, :, 1].mask, f1to2(a[:, 1]).mask)
    assert_array_equal(res[:, :, 2].mask, f1to2(a[:, 2]).mask)