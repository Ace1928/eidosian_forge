import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_prepend_error(self):
    with gzip.open(self.filename, 'wb') as f:
        f.write(data1)
    with gzip.open(self.filename, 'rb') as f:
        f.fileobj.prepend()