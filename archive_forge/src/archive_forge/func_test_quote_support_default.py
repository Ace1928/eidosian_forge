import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_quote_support_default():
    """Support for quoted fields is disabled by default."""
    txt = StringIO('"lat,long", 45, 30\n')
    dtype = np.dtype([('f0', 'U24'), ('f1', np.float64), ('f2', np.float64)])
    with pytest.raises(ValueError, match='the dtype passed requires 3 columns but 4 were'):
        np.loadtxt(txt, dtype=dtype, delimiter=',')
    txt.seek(0)
    expected = np.array([('lat,long', 45.0, 30.0)], dtype=dtype)
    res = np.loadtxt(txt, dtype=dtype, delimiter=',', quotechar='"')
    assert_array_equal(res, expected)