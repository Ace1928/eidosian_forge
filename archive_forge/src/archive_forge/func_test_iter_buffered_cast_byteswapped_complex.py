import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_buffered_cast_byteswapped_complex():
    a = np.arange(10, dtype='c8').newbyteorder().byteswap()
    a += 2j
    i = nditer(a, ['buffered', 'external_loop'], [['readwrite', 'nbo', 'aligned']], casting='same_kind', op_dtypes=[np.dtype('c16')], buffersize=3)
    with i:
        for v in i:
            v[...] *= 2
    assert_equal(a, 2 * np.arange(10, dtype='c8') + 4j)
    a = np.arange(10, dtype='c8')
    a += 2j
    i = nditer(a, ['buffered', 'external_loop'], [['readwrite', 'nbo', 'aligned']], casting='same_kind', op_dtypes=[np.dtype('c16').newbyteorder()], buffersize=3)
    with i:
        for v in i:
            v[...] *= 2
    assert_equal(a, 2 * np.arange(10, dtype='c8') + 4j)
    a = np.arange(10, dtype=np.clongdouble).newbyteorder().byteswap()
    a += 2j
    i = nditer(a, ['buffered', 'external_loop'], [['readwrite', 'nbo', 'aligned']], casting='same_kind', op_dtypes=[np.dtype('c16')], buffersize=3)
    with i:
        for v in i:
            v[...] *= 2
    assert_equal(a, 2 * np.arange(10, dtype=np.clongdouble) + 4j)
    a = np.arange(10, dtype=np.longdouble).newbyteorder().byteswap()
    i = nditer(a, ['buffered', 'external_loop'], [['readwrite', 'nbo', 'aligned']], casting='same_kind', op_dtypes=[np.dtype('f4')], buffersize=7)
    with i:
        for v in i:
            v[...] *= 2
    assert_equal(a, 2 * np.arange(10, dtype=np.longdouble))