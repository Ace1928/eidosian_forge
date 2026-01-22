import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_read_truncated(self):
    data = data1 * 50
    truncated = gzip.compress(data)[:-8]
    with gzip.GzipFile(fileobj=io.BytesIO(truncated)) as f:
        self.assertRaises(EOFError, f.read)
    with gzip.GzipFile(fileobj=io.BytesIO(truncated)) as f:
        self.assertEqual(f.read(len(data)), data)
        self.assertRaises(EOFError, f.read, 1)
    for i in range(2, 10):
        with gzip.GzipFile(fileobj=io.BytesIO(truncated[:i])) as f:
            self.assertRaises(EOFError, f.read, 1)