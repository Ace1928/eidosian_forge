import six
from six.moves import range
import unittest
import os
import io
import struct
from apitools.base.py import gzip
from io import open
def test_fileobj_from_fdopen(self):
    fd = os.open(self.filename, os.O_WRONLY | os.O_CREAT)
    with os.fdopen(fd, 'wb') as f:
        with gzip.GzipFile(fileobj=f, mode='w') as g:
            pass