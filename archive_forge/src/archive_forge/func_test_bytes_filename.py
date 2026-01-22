import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_bytes_filename(self):
    str_filename = self.filename
    try:
        bytes_filename = str_filename.encode('ascii')
    except UnicodeEncodeError:
        self.skipTest('Temporary file name needs to be ASCII')
    with gzip.GzipFile(bytes_filename, 'wb') as f:
        f.write(data1 * 50)
    with gzip.GzipFile(bytes_filename, 'rb') as f:
        self.assertEqual(f.read(), data1 * 50)
    with gzip.GzipFile(str_filename, 'rb') as f:
        self.assertEqual(f.read(), data1 * 50)