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
def testPtrns(self):
    """Test colour type 3 and tRNS chunk (and 4-bit palette)."""
    a = (50, 99, 50, 50)
    b = (200, 120, 120, 80)
    c = (255, 255, 255)
    d = (200, 120, 120)
    e = (50, 99, 50)
    w = Writer(3, 3, bitdepth=4, palette=[a, b, c, d, e])
    f = BytesIO()
    w.write_array(f, array('B', (4, 3, 2, 3, 2, 0, 2, 0, 1)))
    r = Reader(bytes=f.getvalue())
    x, y, pixels, meta = r.asRGBA8()
    self.assertEqual(x, 3)
    self.assertEqual(y, 3)
    c = c + (255,)
    d = d + (255,)
    e = e + (255,)
    boxed = [(e, d, c), (d, c, a), (c, a, b)]
    flat = map(lambda row: itertools.chain(*row), boxed)
    self.assertEqual(map(list, pixels), map(list, flat))