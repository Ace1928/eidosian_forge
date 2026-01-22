import numpy as np
from numpy.testing import (
def test_iterators_exceptions(self):
    """cases in iterators.c"""

    def assign(obj, ind, val):
        obj[ind] = val
    a = np.zeros([1, 2, 3])
    assert_raises(IndexError, lambda: a[0, 5, None, 2])
    assert_raises(IndexError, lambda: a[0, 5, 0, 2])
    assert_raises(IndexError, lambda: assign(a, (0, 5, None, 2), 1))
    assert_raises(IndexError, lambda: assign(a, (0, 5, 0, 2), 1))
    a = np.zeros([1, 0, 3])
    assert_raises(IndexError, lambda: a[0, 0, None, 2])
    assert_raises(IndexError, lambda: assign(a, (0, 0, None, 2), 1))
    a = np.zeros([1, 2, 3])
    assert_raises(IndexError, lambda: a.flat[10])
    assert_raises(IndexError, lambda: assign(a.flat, 10, 5))
    a = np.zeros([1, 0, 3])
    assert_raises(IndexError, lambda: a.flat[10])
    assert_raises(IndexError, lambda: assign(a.flat, 10, 5))
    a = np.zeros([1, 2, 3])
    assert_raises(IndexError, lambda: a.flat[np.array(10)])
    assert_raises(IndexError, lambda: assign(a.flat, np.array(10), 5))
    a = np.zeros([1, 0, 3])
    assert_raises(IndexError, lambda: a.flat[np.array(10)])
    assert_raises(IndexError, lambda: assign(a.flat, np.array(10), 5))
    a = np.zeros([1, 2, 3])
    assert_raises(IndexError, lambda: a.flat[np.array([10])])
    assert_raises(IndexError, lambda: assign(a.flat, np.array([10]), 5))
    a = np.zeros([1, 0, 3])
    assert_raises(IndexError, lambda: a.flat[np.array([10])])
    assert_raises(IndexError, lambda: assign(a.flat, np.array([10]), 5))