import os
import zlib
from io import BytesIO
from tempfile import mkstemp
from contextlib import contextmanager
import numpy as np
from numpy.testing import assert_, assert_equal
from pytest import raises as assert_raises
from scipy.io.matlab._streams import (make_stream,
def test_all_data_read(self):
    compressed_stream, compressed_data_len, data = self._get_data(1024)
    stream = ZlibInputStream(compressed_stream, compressed_data_len)
    assert_(not stream.all_data_read())
    stream.seek(512)
    assert_(not stream.all_data_read())
    stream.seek(1024)
    assert_(stream.all_data_read())