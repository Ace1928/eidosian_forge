import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
def test_object_cleanup_on_read_error():
    sentinel = object()
    already_read = 0

    def conv(x):
        nonlocal already_read
        if already_read > 4999:
            raise ValueError('failed half-way through!')
        already_read += 1
        return sentinel
    txt = StringIO('x\n' * 10000)
    with pytest.raises(ValueError, match='at row 5000, column 1'):
        np.loadtxt(txt, dtype=object, converters={0: conv})
    assert sys.getrefcount(sentinel) == 2