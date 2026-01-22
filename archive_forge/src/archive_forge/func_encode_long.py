from types import FunctionType
from copyreg import dispatch_table
from copyreg import _extension_registry, _inverted_registry, _extension_cache
from itertools import islice
from functools import partial
import sys
from sys import maxsize
from struct import pack, unpack
import re
import io
import codecs
import _compat_pickle
def encode_long(x):
    """Encode a long to a two's complement little-endian binary string.
    Note that 0 is a special case, returning an empty string, to save a
    byte in the LONG1 pickling context.

    >>> encode_long(0)
    b''
    >>> encode_long(255)
    b'\\xff\\x00'
    >>> encode_long(32767)
    b'\\xff\\x7f'
    >>> encode_long(-256)
    b'\\x00\\xff'
    >>> encode_long(-32768)
    b'\\x00\\x80'
    >>> encode_long(-128)
    b'\\x80'
    >>> encode_long(127)
    b'\\x7f'
    >>>
    """
    if x == 0:
        return b''
    nbytes = (x.bit_length() >> 3) + 1
    result = x.to_bytes(nbytes, byteorder='little', signed=True)
    if x < 0 and nbytes > 1:
        if result[-1] == 255 and result[-2] & 128 != 0:
            result = result[:-1]
    return result