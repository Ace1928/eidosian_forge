import os
import zlib
from io import BytesIO
from tempfile import mkstemp
from contextlib import contextmanager
import numpy as np
from numpy.testing import assert_, assert_equal
from pytest import raises as assert_raises
from scipy.io.matlab._streams import (make_stream,
def test_seek(self):
    compressed_stream, compressed_data_len, data = self._get_data(1024)
    stream = ZlibInputStream(compressed_stream, compressed_data_len)
    stream.seek(123)
    p = 123
    assert_equal(stream.tell(), p)
    d1 = stream.read(11)
    assert_equal(d1, data[p:p + 11])
    stream.seek(321, 1)
    p = 123 + 11 + 321
    assert_equal(stream.tell(), p)
    d2 = stream.read(21)
    assert_equal(d2, data[p:p + 21])
    stream.seek(641, 0)
    p = 641
    assert_equal(stream.tell(), p)
    d3 = stream.read(11)
    assert_equal(d3, data[p:p + 11])
    assert_raises(OSError, stream.seek, 10, 2)
    assert_raises(OSError, stream.seek, -1, 1)
    assert_raises(ValueError, stream.seek, 1, 123)
    stream.seek(10000, 1)
    assert_raises(OSError, stream.read, 12)