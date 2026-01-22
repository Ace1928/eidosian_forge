import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_with_open(self):
    with gzip.GzipFile(self.filename, 'wb') as f:
        f.write(b'xxx')
    f = gzip.GzipFile(self.filename, 'rb')
    f.close()
    try:
        with f:
            pass
    except ValueError:
        pass
    else:
        self.fail("__enter__ on a closed file didn't raise an exception")
    try:
        with gzip.GzipFile(self.filename, 'wb') as f:
            1 / 0
    except ZeroDivisionError:
        pass
    else:
        self.fail("1/0 didn't raise an exception")