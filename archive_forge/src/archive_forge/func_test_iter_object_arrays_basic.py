import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_object_arrays_basic():
    obj = {'a': 3, 'b': 'd'}
    a = np.array([[1, 2, 3], None, obj, None], dtype='O')
    if HAS_REFCOUNT:
        rc = sys.getrefcount(obj)
    assert_raises(TypeError, nditer, a)
    if HAS_REFCOUNT:
        assert_equal(sys.getrefcount(obj), rc)
    i = nditer(a, ['refs_ok'], ['readonly'])
    vals = [x_[()] for x_ in i]
    assert_equal(np.array(vals, dtype='O'), a)
    vals, i, x = [None] * 3
    if HAS_REFCOUNT:
        assert_equal(sys.getrefcount(obj), rc)
    i = nditer(a.reshape(2, 2).T, ['refs_ok', 'buffered'], ['readonly'], order='C')
    assert_(i.iterationneedsapi)
    vals = [x_[()] for x_ in i]
    assert_equal(np.array(vals, dtype='O'), a.reshape(2, 2).ravel(order='F'))
    vals, i, x = [None] * 3
    if HAS_REFCOUNT:
        assert_equal(sys.getrefcount(obj), rc)
    i = nditer(a.reshape(2, 2).T, ['refs_ok', 'buffered'], ['readwrite'], order='C')
    with i:
        for x in i:
            x[...] = None
        vals, i, x = [None] * 3
    if HAS_REFCOUNT:
        assert_(sys.getrefcount(obj) == rc - 1)
    assert_equal(a, np.array([None] * 4, dtype='O'))