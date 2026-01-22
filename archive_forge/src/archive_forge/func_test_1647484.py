import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_1647484(self):
    for mode in ('wb', 'rb'):
        with gzip.GzipFile(self.filename, mode) as f:
            self.assertTrue(hasattr(f, 'name'))
            self.assertEqual(f.name, self.filename)