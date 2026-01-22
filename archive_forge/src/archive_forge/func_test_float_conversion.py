import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_float_conversion():
    """
    Some tests that the conversion to float64 works as accurately as the
    Python built-in `float` function. In a naive version of the float parser,
    these strings resulted in values that were off by an ULP or two.
    """
    strings = ['0.9999999999999999', '9876543210.123456', '5.43215432154321e+300', '0.901', '0.333']
    txt = StringIO('\n'.join(strings))
    res = np.loadtxt(txt)
    expected = np.array([float(s) for s in strings])
    assert_equal(res, expected)