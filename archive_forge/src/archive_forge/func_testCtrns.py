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
def testCtrns(self):
    """Test colour type 2 and tRNS chunk."""
    r = Reader(bytes=_pngsuite['tbrn2c08'])
    x, y, pixels, meta = r.asRGBA8()
    row0 = list(pixels)[0]
    self.assertEqual(tuple(row0[0:4]), (127, 127, 127, 0))