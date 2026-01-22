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
def check_sizes(size, width, height):
    """
    Check that these arguments, if supplied, are consistent.
    Return a (width, height) pair.
    """
    if not size:
        return (width, height)
    if len(size) != 2:
        raise ProtocolError('size argument should be a pair (width, height)')
    if width is not None and width != size[0]:
        raise ProtocolError(f'size[0] ({size[0]}) and width ({width}) should match when both are used.')
    if height is not None and height != size[1]:
        raise ProtocolError(f'size[1] ({size[1]}) and height ({height}) should match when both are used.')
    return size