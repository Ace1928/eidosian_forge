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
def iterboxed(self, rows):
    """Iterator that yields each scanline in boxed row flat pixel
        format.  `rows` should be an iterator that yields the bytes of
        each row in turn.
        """

    def asvalues(raw):
        """Convert a row of raw bytes into a flat row.  Result may
            or may not share with argument"""
        if self.bitdepth == 8:
            return raw
        if self.bitdepth == 16:
            raw = tostring(raw)
            return array('H', struct.unpack('!%dH' % (len(raw) // 2), raw))
        assert self.bitdepth < 8
        width = self.width
        spb = 8 // self.bitdepth
        out = array('B')
        mask = 2 ** self.bitdepth - 1
        shifts = map(self.bitdepth.__mul__, reversed(range(spb)))
        for o in raw:
            out.extend(map(lambda i: mask & o >> i, shifts))
        return out[:width]
    return map(asvalues, rows)