import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_allocated_array_dtypes():
    it = np.nditer(([1, 3, 20], None), op_dtypes=[None, ('i4', (2,))])
    for a, b in it:
        b[0] = a - 1
        b[1] = a + 1
    assert_equal(it.operands[1], [[0, 2], [2, 4], [19, 21]])
    it = np.nditer(([[1, 3, 20]], None), op_dtypes=[None, ('i4', (2,))], flags=['reduce_ok'], op_axes=[None, (-1, 0)])
    for a, b in it:
        b[0] = a - 1
        b[1] = a + 1
    assert_equal(it.operands[1], [[0, 2], [2, 4], [19, 21]])
    it = np.nditer((10, 2, None), op_dtypes=[None, None, ('i4', (2, 2))])
    for a, b, c in it:
        c[0, 0] = a - b
        c[0, 1] = a + b
        c[1, 0] = a * b
        c[1, 1] = a / b
    assert_equal(it.operands[2], [[8, 12], [20, 5]])