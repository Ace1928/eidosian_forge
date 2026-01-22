import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_copy_casts_structured():
    in_dtype = np.dtype([('a', np.dtype('i,')), ('b', np.dtype('>i,<i,>d,S17,>d,(3)f,O,i1'))])
    out_dtype = np.dtype([('a', np.dtype('O')), ('b', np.dtype('>i,>i,S17,>d,>U3,(3)d,i1,O'))])
    arr = np.ones(1000, dtype=in_dtype)
    it = np.nditer((arr,), ['buffered', 'external_loop', 'refs_ok'], op_dtypes=[out_dtype], casting='unsafe')
    it_copy = it.copy()
    res1 = next(it)
    del it
    res2 = next(it_copy)
    del it_copy
    expected = arr['a'].astype(out_dtype['a'])
    assert_array_equal(res1['a'], expected)
    assert_array_equal(res2['a'], expected)
    for field in in_dtype['b'].names:
        expected = arr['b'][field].astype(out_dtype['b'][field].base)
        assert_array_equal(res1['b'][field], expected)
        assert_array_equal(res2['b'][field], expected)