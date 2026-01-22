import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_peek(self):
    uncompressed = data1 * 200
    with gzip.GzipFile(self.filename, 'wb') as f:
        f.write(uncompressed)

    def sizes():
        while True:
            for n in range(5, 50, 10):
                yield n
    with gzip.GzipFile(self.filename, 'rb') as f:
        f.max_read_chunk = 33
        nread = 0
        for n in sizes():
            s = f.peek(n)
            if s == b'':
                break
            self.assertEqual(f.read(len(s)), s)
            nread += len(s)
        self.assertEqual(f.read(100), b'')
        self.assertEqual(nread, len(uncompressed))