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
def testfromarray(self):
    img = from_array([[0, 51, 102], [255, 204, 153]], 'L')
    img.save('testfromarray.png')