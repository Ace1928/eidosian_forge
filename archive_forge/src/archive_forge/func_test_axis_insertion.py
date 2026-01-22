import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_axis_insertion(self, cls=np.ndarray):

    def f1to2(x):
        """produces an asymmetric non-square matrix from x"""
        assert_equal(x.ndim, 1)
        return (x[::-1] * x[1:, None]).view(cls)
    a2d = np.arange(6 * 3).reshape((6, 3))
    actual = apply_along_axis(f1to2, 0, a2d)
    expected = np.stack([f1to2(a2d[:, i]) for i in range(a2d.shape[1])], axis=-1).view(cls)
    assert_equal(type(actual), type(expected))
    assert_equal(actual, expected)
    actual = apply_along_axis(f1to2, 1, a2d)
    expected = np.stack([f1to2(a2d[i, :]) for i in range(a2d.shape[0])], axis=0).view(cls)
    assert_equal(type(actual), type(expected))
    assert_equal(actual, expected)
    a3d = np.arange(6 * 5 * 3).reshape((6, 5, 3))
    actual = apply_along_axis(f1to2, 1, a3d)
    expected = np.stack([np.stack([f1to2(a3d[i, :, j]) for i in range(a3d.shape[0])], axis=0) for j in range(a3d.shape[2])], axis=-1).view(cls)
    assert_equal(type(actual), type(expected))
    assert_equal(actual, expected)