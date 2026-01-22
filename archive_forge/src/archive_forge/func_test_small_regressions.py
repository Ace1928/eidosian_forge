import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_small_regressions(self):
    a = np.array([0])
    if HAS_REFCOUNT:
        refcount = sys.getrefcount(np.dtype(np.intp))
    a[np.array([0], dtype=np.intp)] = 1
    a[np.array([0], dtype=np.uint8)] = 1
    assert_raises(IndexError, a.__setitem__, np.array([1], dtype=np.intp), 1)
    assert_raises(IndexError, a.__setitem__, np.array([1], dtype=np.uint8), 1)
    if HAS_REFCOUNT:
        assert_equal(sys.getrefcount(np.dtype(np.intp)), refcount)