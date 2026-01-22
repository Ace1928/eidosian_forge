import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('dtype', (float, object))
def test_maxrows_no_blank_lines(dtype):
    txt = StringIO('1.5,2.5\n3.0,4.0\n5.5,6.0')
    res = np.loadtxt(txt, dtype=dtype, delimiter=',', max_rows=2)
    assert_equal(res.dtype, dtype)
    assert_equal(res, np.array([['1.5', '2.5'], ['3.0', '4.0']], dtype=dtype))