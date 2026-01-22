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
def topngbytes(name, rows, x, y, **k):
    """Convenience function for creating a PNG file "in memory" as a
    string.  Creates a :class:`Writer` instance using the keyword arguments,
    then passes `rows` to its :meth:`Writer.write` method.  The resulting
    PNG file is returned as a string.  `name` is used to identify the file for
    debugging.
    """
    import os
    print(name)
    f = BytesIO()
    w = Writer(x, y, **k)
    w.write(f, rows)
    if os.environ.get('PYPNG_TEST_TMP'):
        w = open(name, 'wb')
        w.write(f.getvalue())
        w.close()
    return f.getvalue()