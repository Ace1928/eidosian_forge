import collections
import io   # For io.BytesIO
import itertools
import math
import operator
import re
import struct
import sys
import warnings
import zlib
from array import array
fromarray = from_array
def read_flat(self):
    """
        Read a PNG file and decode it into a single array of values.
        Returns (*width*, *height*, *values*, *info*).

        May use excessive memory.

        `values` is a single array.

        The :meth:`read` method is more stream-friendly than this,
        because it returns a sequence of rows.
        """
    x, y, pixel, info = self.read()
    arraycode = 'BH'[info['bitdepth'] > 8]
    pixel = array(arraycode, itertools.chain(*pixel))
    return (x, y, pixel, info)