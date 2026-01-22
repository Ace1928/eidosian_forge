import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_unicode_with_converter():
    txt = StringIO('cat,dog\nαβγ,δεζ\nabc,def\n')
    conv = {0: lambda s: s.upper()}
    res = np.loadtxt(txt, dtype=np.dtype('U12'), converters=conv, delimiter=',', encoding=None)
    expected = np.array([['CAT', 'dog'], ['ΑΒΓ', 'δεζ'], ['ABC', 'def']])
    assert_equal(res, expected)