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
def testWinfo(self):
    """Test the dictionary returned by a `read` method can be used
        as args for :meth:`Writer`.
        """
    r = Reader(bytes=_pngsuite['basn2c16'])
    info = r.read()[3]
    w = Writer(**info)