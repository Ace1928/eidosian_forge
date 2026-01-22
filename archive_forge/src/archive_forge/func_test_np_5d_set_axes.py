from scipy.fft._helper import next_fast_len, _init_nd_shape_and_axes
from numpy.testing import assert_equal, assert_array_equal
from pytest import raises as assert_raises
import pytest
import numpy as np
import sys
def test_np_5d_set_axes(self):
    x = np.zeros([6, 2, 5, 3, 4])
    shape = None
    axes = [4, 1, 2]
    shape_expected = np.array([4, 2, 5])
    axes_expected = np.array([4, 1, 2])
    shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)
    assert_equal(shape_res, shape_expected)
    assert_equal(axes_res, axes_expected)