from the public API.  This format is called packed.  When packed,
import io
import itertools
import math
import operator
import struct
import sys
import zlib
import warnings
from array import array
from functools import reduce
from pygame.tests.test_utils import tostring
fromarray = from_array
import tempfile
import unittest
def testPNMsbit(self):
    """Test that PNM files can generates sBIT chunk."""

    def do():
        return _main(['testPNMsbit'])
    s = BytesIO()
    s.write(strtobytes('P6 8 1 1\n'))
    for pixel in range(8):
        s.write(struct.pack('<I', 16513 * pixel & 65793)[:3])
    s.flush()
    s.seek(0)
    o = BytesIO()
    testWithIO(s, o, do)
    r = Reader(bytes=o.getvalue())
    sbit = r.chunk('sBIT')[1]
    self.assertEqual(sbit, strtobytes('\x01\x01\x01'))