import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_io_on_closed_object(self):
    self.test_write()
    f = gzip.GzipFile(self.filename, 'r')
    f.close()
    with self.assertRaises(ValueError):
        f.read(1)
    with self.assertRaises(ValueError):
        f.seek(0)
    with self.assertRaises(ValueError):
        f.tell()
    f = gzip.GzipFile(self.filename, 'w')
    f.close()
    with self.assertRaises(ValueError):
        f.write(b'')
    with self.assertRaises(ValueError):
        f.flush()