import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_dtype_copy(self):
    a = arange(6, dtype='i4').reshape(2, 3)
    i, j = np.nested_iters(a, [[0], [1]], op_flags=['readonly', 'copy'], op_dtypes='f8')
    assert_equal(j[0].dtype, np.dtype('f8'))
    vals = [list(j) for _ in i]
    assert_equal(vals, [[0, 1, 2], [3, 4, 5]])
    vals = None
    a = arange(6, dtype='f4').reshape(2, 3)
    i, j = np.nested_iters(a, [[0], [1]], op_flags=['readwrite', 'updateifcopy'], casting='same_kind', op_dtypes='f8')
    with i, j:
        assert_equal(j[0].dtype, np.dtype('f8'))
        for x in i:
            for y in j:
                y[...] += 1
        assert_equal(a, [[0, 1, 2], [3, 4, 5]])
    assert_equal(a, [[1, 2, 3], [4, 5, 6]])
    a = arange(6, dtype='f4').reshape(2, 3)
    i, j = np.nested_iters(a, [[0], [1]], op_flags=['readwrite', 'updateifcopy'], casting='same_kind', op_dtypes='f8')
    assert_equal(j[0].dtype, np.dtype('f8'))
    for x in i:
        for y in j:
            y[...] += 1
    assert_equal(a, [[0, 1, 2], [3, 4, 5]])
    i.close()
    j.close()
    assert_equal(a, [[1, 2, 3], [4, 5, 6]])