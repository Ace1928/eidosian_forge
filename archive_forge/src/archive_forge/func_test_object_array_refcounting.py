import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_object_array_refcounting(self):
    if not hasattr(sys, 'getrefcount'):
        return
    cnt = sys.getrefcount
    a = object()
    b = object()
    c = object()
    cnt0_a = cnt(a)
    cnt0_b = cnt(b)
    cnt0_c = cnt(c)
    arr = np.zeros(5, dtype=np.object_)
    arr[:] = a
    assert_equal(cnt(a), cnt0_a + 5)
    arr[:] = b
    assert_equal(cnt(a), cnt0_a)
    assert_equal(cnt(b), cnt0_b + 5)
    arr[:2] = c
    assert_equal(cnt(b), cnt0_b + 3)
    assert_equal(cnt(c), cnt0_c + 2)
    del arr
    arr = np.zeros((5, 2), dtype=np.object_)
    arr0 = np.zeros(2, dtype=np.object_)
    arr0[0] = a
    assert_(cnt(a) == cnt0_a + 1)
    arr0[1] = b
    assert_(cnt(b) == cnt0_b + 1)
    arr[:, :] = arr0
    assert_(cnt(a) == cnt0_a + 6)
    assert_(cnt(b) == cnt0_b + 6)
    arr[:, 0] = None
    assert_(cnt(a) == cnt0_a + 1)
    del arr, arr0
    arr = np.zeros((5, 2), dtype=np.object_)
    arr[:, 0] = a
    arr[:, 1] = b
    assert_(cnt(a) == cnt0_a + 5)
    assert_(cnt(b) == cnt0_b + 5)
    arr2 = arr.copy()
    assert_(cnt(a) == cnt0_a + 10)
    assert_(cnt(b) == cnt0_b + 10)
    arr2 = arr[:, 0].copy()
    assert_(cnt(a) == cnt0_a + 10)
    assert_(cnt(b) == cnt0_b + 5)
    arr2 = arr.flatten()
    assert_(cnt(a) == cnt0_a + 10)
    assert_(cnt(b) == cnt0_b + 10)
    del arr, arr2
    arr1 = np.zeros((5, 1), dtype=np.object_)
    arr2 = np.zeros((5, 1), dtype=np.object_)
    arr1[...] = a
    arr2[...] = b
    assert_(cnt(a) == cnt0_a + 5)
    assert_(cnt(b) == cnt0_b + 5)
    tmp = np.concatenate((arr1, arr2))
    assert_(cnt(a) == cnt0_a + 5 + 5)
    assert_(cnt(b) == cnt0_b + 5 + 5)
    tmp = arr1.repeat(3, axis=0)
    assert_(cnt(a) == cnt0_a + 5 + 3 * 5)
    tmp = arr1.take([1, 2, 3], axis=0)
    assert_(cnt(a) == cnt0_a + 5 + 3)
    x = np.array([[0], [1], [0], [1], [1]], int)
    tmp = x.choose(arr1, arr2)
    assert_(cnt(a) == cnt0_a + 5 + 2)
    assert_(cnt(b) == cnt0_b + 5 + 3)
    del tmp