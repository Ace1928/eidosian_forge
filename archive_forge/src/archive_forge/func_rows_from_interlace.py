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
def rows_from_interlace():
    """Yield each row from an interlaced PNG."""
    bs = bytearray(itertools.chain(*raw))
    arraycode = 'BH'[self.bitdepth > 8]
    values = self._deinterlace(bs)
    vpr = self.width * self.planes
    for i in range(0, len(values), vpr):
        row = array(arraycode, values[i:i + vpr])
        yield row