import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
@pytest.mark.skipif(sys.version_info[:2] == (3, 9) and sys.platform == 'win32', reason='Errors with Python 3.9 on Windows')
@pytest.mark.parametrize(['in_dtype', 'buf_dtype'], [('i', 'O'), ('O', 'i'), ('i,O', 'O,O'), ('O,i', 'i,O')])
@pytest.mark.parametrize('steps', [1, 2, 3])
def test_partial_iteration_cleanup(in_dtype, buf_dtype, steps):
    """
    Checks for reference counting leaks during cleanup.  Using explicit
    reference counts lead to occasional false positives (at least in parallel
    test setups).  This test now should still test leaks correctly when
    run e.g. with pytest-valgrind or pytest-leaks
    """
    value = 2 ** 30 + 1
    arr = np.full(int(np.BUFSIZE * 2.5), value).astype(in_dtype)
    it = np.nditer(arr, op_dtypes=[np.dtype(buf_dtype)], flags=['buffered', 'external_loop', 'refs_ok'], casting='unsafe')
    for step in range(steps):
        next(it)
    del it
    it = np.nditer(arr, op_dtypes=[np.dtype(buf_dtype)], flags=['buffered', 'external_loop', 'refs_ok'], casting='unsafe')
    for step in range(steps):
        it.iternext()
    del it