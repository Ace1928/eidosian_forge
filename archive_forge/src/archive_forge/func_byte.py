from __future__ import absolute_import
import itertools
import sys
from struct import pack
def byte(num):
    """
    Converts a number between 0 and 255 (both inclusive) to a base-256 (byte)
    representation.

    Use it as a replacement for ``chr`` where you are expecting a byte
    because this will work on all current versions of Python::

    :param num:
        An unsigned integer between 0 and 255 (both inclusive).
    :returns:
        A single byte.
    """
    return pack('B', num)