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
def testBadFilter(self):

    def eachchunk(chunk):
        if chunk[0] != 'IDAT':
            return chunk
        data = zlib.decompress(chunk[1])
        data = strtobytes('\x99') + data[1:]
        data = zlib.compress(data)
        return (chunk[0], data)
    self.assertRaises(FormatError, self.helperFormat, eachchunk)