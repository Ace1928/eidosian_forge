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
def testfromarrayIter(self):
    import itertools
    i = itertools.islice(itertools.count(10), 20)
    i = map(lambda x: [x, x, x], i)
    img = from_array(i, 'RGB;5', dict(height=20))
    f = open('testiter.png', 'wb')
    img.save(f)
    f.close()