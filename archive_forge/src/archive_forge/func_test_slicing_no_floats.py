import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_slicing_no_floats(self):
    a = np.array([[5]])
    assert_raises(TypeError, lambda: a[0.0:])
    assert_raises(TypeError, lambda: a[0:, 0.0:2])
    assert_raises(TypeError, lambda: a[0.0::2, :0])
    assert_raises(TypeError, lambda: a[0.0:1:2, :])
    assert_raises(TypeError, lambda: a[:, 0.0:])
    assert_raises(TypeError, lambda: a[:0.0])
    assert_raises(TypeError, lambda: a[:0, 1:2.0])
    assert_raises(TypeError, lambda: a[:0.0:2, :0])
    assert_raises(TypeError, lambda: a[:0.0, :])
    assert_raises(TypeError, lambda: a[:, 0:4.0:2])
    assert_raises(TypeError, lambda: a[::1.0])
    assert_raises(TypeError, lambda: a[0:, :2:2.0])
    assert_raises(TypeError, lambda: a[1::4.0, :0])
    assert_raises(TypeError, lambda: a[::5.0, :])
    assert_raises(TypeError, lambda: a[:, 0:4:2.0])
    assert_raises(TypeError, lambda: a[1.0:2:2.0])
    assert_raises(TypeError, lambda: a[1.0::2.0])
    assert_raises(TypeError, lambda: a[0:, :2.0:2.0])
    assert_raises(TypeError, lambda: a[1.0:1:4.0, :0])
    assert_raises(TypeError, lambda: a[1.0:5.0:5.0, :])
    assert_raises(TypeError, lambda: a[:, 0.4:4.0:2.0])
    assert_raises(TypeError, lambda: a[::0.0])