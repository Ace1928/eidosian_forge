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
def undo_filter_sub(filter_unit, scanline, previous, result):
    """Undo sub filter."""
    ai = 0
    for i in range(filter_unit, len(result)):
        x = scanline[i]
        a = result[ai]
        result[i] = x + a & 255
        ai += 1