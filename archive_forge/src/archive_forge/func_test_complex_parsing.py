import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('dtype', (np.complex64, np.complex128))
@pytest.mark.parametrize('with_parens', (False, True))
def test_complex_parsing(dtype, with_parens):
    s = '(1.0-2.5j),3.75,(7+-5.0j)\n(4),(-19e2j),(0)'
    if not with_parens:
        s = s.replace('(', '').replace(')', '')
    res = np.loadtxt(StringIO(s), dtype=dtype, delimiter=',')
    expected = np.array([[1.0 - 2.5j, 3.75, 7 - 5j], [4.0, -1900j, 0]], dtype=dtype)
    assert_equal(res, expected)