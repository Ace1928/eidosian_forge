import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_object_arrays_conversions():
    a = np.arange(6, dtype='O')
    i = nditer(a, ['refs_ok', 'buffered'], ['readwrite'], casting='unsafe', op_dtypes='i4')
    with i:
        for x in i:
            x[...] += 1
    assert_equal(a, np.arange(6) + 1)
    a = np.arange(6, dtype='i4')
    i = nditer(a, ['refs_ok', 'buffered'], ['readwrite'], casting='unsafe', op_dtypes='O')
    with i:
        for x in i:
            x[...] += 1
    assert_equal(a, np.arange(6) + 1)
    a = np.zeros((6,), dtype=[('p', 'i1'), ('a', 'O')])
    a = a['a']
    a[:] = np.arange(6)
    i = nditer(a, ['refs_ok', 'buffered'], ['readwrite'], casting='unsafe', op_dtypes='i4')
    with i:
        for x in i:
            x[...] += 1
    assert_equal(a, np.arange(6) + 1)
    a = np.zeros((6,), dtype=[('p', 'i1'), ('a', 'i4')])
    a = a['a']
    a[:] = np.arange(6) + 98172488
    i = nditer(a, ['refs_ok', 'buffered'], ['readwrite'], casting='unsafe', op_dtypes='O')
    with i:
        ob = i[0][()]
        if HAS_REFCOUNT:
            rc = sys.getrefcount(ob)
        for x in i:
            x[...] += 1
    if HAS_REFCOUNT:
        assert_(sys.getrefcount(ob) == rc - 1)
    assert_equal(a, np.arange(6) + 98172489)