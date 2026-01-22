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
def testAdam7read(self):
    """Adam7 interlace reading.
        Specifically, test that for images in the PngSuite that
        have both an interlaced and straightlaced pair that both
        images from the pair produce the same array of pixels."""
    for candidate in _pngsuite:
        if not candidate.startswith('basn'):
            continue
        candi = candidate.replace('n', 'i')
        if candi not in _pngsuite:
            continue
        print(f'adam7 read {candidate}')
        straight = Reader(bytes=_pngsuite[candidate])
        adam7 = Reader(bytes=_pngsuite[candi])
        straight = straight.read()[2]
        adam7 = adam7.read()[2]
        self.assertEqual(map(list, straight), map(list, adam7))