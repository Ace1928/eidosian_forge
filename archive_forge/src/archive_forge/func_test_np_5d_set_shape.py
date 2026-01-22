from scipy.fft._helper import next_fast_len, _init_nd_shape_and_axes
from numpy.testing import assert_equal, assert_array_equal
from pytest import raises as assert_raises
import pytest
import numpy as np
import sys
def test_np_5d_set_shape(self):
    x = np.zeros([6, 2, 5, 3, 4])
    shape = [10, -1, -1, 1, 4]
    axes = None
    shape_expected = np.array([10, 2, 5, 1, 4])
    axes_expected = np.array([0, 1, 2, 3, 4])
    shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)
    assert_equal(shape_res, shape_expected)
    assert_equal(axes_res, axes_expected)