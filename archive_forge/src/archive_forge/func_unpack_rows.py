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
def unpack_rows(rows):
    """Unpack each row from being 16-bits per value,
    to being a sequence of bytes.
    """
    for row in rows:
        fmt = f'!{len(row)}H'
        yield bytearray(struct.pack(fmt, *row))