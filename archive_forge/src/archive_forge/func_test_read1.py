import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_read1(self):
    self.test_write()
    blocks = []
    nread = 0
    with gzip.GzipFile(self.filename, 'r') as f:
        while True:
            d = f.read1()
            if not d:
                break
            blocks.append(d)
            nread += len(d)
            self.assertEqual(f.tell(), nread)
    self.assertEqual(b''.join(blocks), data1 * 50)