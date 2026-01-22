import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
@pytest.mark.parametrize('arr', [np.ones((8000, 4, 2), dtype=object)[:, ::2, :], np.ones((8000, 4, 2), dtype=object, order='F')[:, ::2, :], np.ones((8000, 4, 2), dtype=object)[:, ::2, :].copy('F')])
def test_object_iter_cleanup_large_reduce(arr):
    out = np.ones(8000, dtype=np.intp)
    res = np.sum(arr, axis=(1, 2), dtype=object, out=out)
    assert_array_equal(res, np.full(8000, 4, dtype=object))