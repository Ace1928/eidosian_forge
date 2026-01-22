import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize(('given_dtype', 'expected_dtype'), [('S', np.dtype('S5')), ('U', np.dtype('U5'))])
def test_string_no_length_given(given_dtype, expected_dtype):
    """
    The given dtype is just 'S' or 'U' with no length. In these cases, the
    length of the resulting dtype is determined by the longest string found
    in the file.
    """
    txt = StringIO('AAA,5-1\nBBBBB,0-3\nC,4-9\n')
    res = np.loadtxt(txt, dtype=given_dtype, delimiter=',')
    expected = np.array([['AAA', '5-1'], ['BBBBB', '0-3'], ['C', '4-9']], dtype=expected_dtype)
    assert_equal(res, expected)
    assert_equal(res.dtype, expected_dtype)