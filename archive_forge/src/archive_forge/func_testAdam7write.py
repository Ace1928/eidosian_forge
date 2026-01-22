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
def testAdam7write(self):
    """Adam7 interlace writing.
        For each test image in the PngSuite, write an interlaced
        and a straightlaced version.  Decode both, and compare results.
        """
    for name, bytes in _pngsuite.items():
        if name[3:5] not in ['n0', 'n2', 'n4', 'n6']:
            continue
        it = Reader(bytes=bytes)
        x, y, pixels, meta = it.read()
        pngi = topngbytes(f'adam7wn{name}.png', pixels, x=x, y=y, bitdepth=it.bitdepth, greyscale=it.greyscale, alpha=it.alpha, transparent=it.transparent, interlace=False)
        x, y, ps, meta = Reader(bytes=pngi).read()
        it = Reader(bytes=bytes)
        x, y, pixels, meta = it.read()
        pngs = topngbytes(f'adam7wi{name}.png', pixels, x=x, y=y, bitdepth=it.bitdepth, greyscale=it.greyscale, alpha=it.alpha, transparent=it.transparent, interlace=True)
        x, y, pi, meta = Reader(bytes=pngs).read()
        self.assertEqual(map(list, ps), map(list, pi))