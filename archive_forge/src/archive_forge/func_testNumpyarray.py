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
def testNumpyarray(self):
    """numpy array."""
    try:
        import numpy
    except ImportError:
        sys.stderr.write('skipping numpy test\n')
        return
    pixels = numpy.array([[0, 21845], [21845, 43690]], numpy.uint16)
    img = from_array(pixels, 'L')
    img.save('testnumpyL16.png')