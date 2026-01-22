import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
@pytest.mark.parametrize(['in_dtype', 'buf_dtype'], [('O', 'i'), ('O,i', 'i,O')])
def test_partial_iteration_error(in_dtype, buf_dtype):
    value = 123
    arr = np.full(int(np.BUFSIZE * 2.5), value).astype(in_dtype)
    if in_dtype == 'O':
        arr[int(np.BUFSIZE * 1.5)] = None
    else:
        arr[int(np.BUFSIZE * 1.5)]['f0'] = None
    count = sys.getrefcount(value)
    it = np.nditer(arr, op_dtypes=[np.dtype(buf_dtype)], flags=['buffered', 'external_loop', 'refs_ok'], casting='unsafe')
    with pytest.raises(TypeError):
        next(it)
        next(it)
    it.reset()
    with pytest.raises(TypeError):
        it.iternext()
        it.iternext()
    assert count == sys.getrefcount(value)