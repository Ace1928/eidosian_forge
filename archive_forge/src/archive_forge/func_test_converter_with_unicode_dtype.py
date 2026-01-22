import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_converter_with_unicode_dtype():
    """
    With the default 'bytes' encoding, tokens are encoded prior to being
    passed to the converter. This means that the output of the converter may
    be bytes instead of unicode as expected by `read_rows`.

    This test checks that outputs from the above scenario are properly decoded
    prior to parsing by `read_rows`.
    """
    txt = StringIO('abc,def\nrst,xyz')
    conv = bytes.upper
    res = np.loadtxt(txt, dtype=np.dtype('U3'), converters=conv, delimiter=',')
    expected = np.array([['ABC', 'DEF'], ['RST', 'XYZ']])
    assert_equal(res, expected)