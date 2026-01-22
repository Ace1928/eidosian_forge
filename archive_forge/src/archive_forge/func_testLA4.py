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
def testLA4(self):
    """Create an LA image with bitdepth 4."""
    bytes = topngbytes('la4.png', [[5, 12]], 1, 1, greyscale=True, alpha=True, bitdepth=4)
    sbit = Reader(bytes=bytes).chunk('sBIT')[1]
    self.assertEqual(sbit, strtobytes('\x04\x04'))