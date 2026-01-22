import numpy as np
from numpy.testing import assert_array_equal
from pytest import raises as assert_raises
from scipy.signal._arraytools import (axis_slice, axis_reverse,
def test_axis_slice(self):
    a = np.arange(12).reshape(3, 4)
    s = axis_slice(a, start=0, stop=1, axis=0)
    assert_array_equal(s, a[0:1, :])
    s = axis_slice(a, start=-1, axis=0)
    assert_array_equal(s, a[-1:, :])
    s = axis_slice(a, start=0, stop=1, axis=1)
    assert_array_equal(s, a[:, 0:1])
    s = axis_slice(a, start=-1, axis=1)
    assert_array_equal(s, a[:, -1:])
    s = axis_slice(a, start=0, step=2, axis=0)
    assert_array_equal(s, a[::2, :])
    s = axis_slice(a, start=0, step=2, axis=1)
    assert_array_equal(s, a[:, ::2])