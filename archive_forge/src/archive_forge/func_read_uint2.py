import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_uint2(f):
    """
    >>> import io
    >>> read_uint2(io.BytesIO(b'\\xff\\x00'))
    255
    >>> read_uint2(io.BytesIO(b'\\xff\\xff'))
    65535
    """
    data = f.read(2)
    if len(data) == 2:
        return _unpack('<H', data)[0]
    raise ValueError('not enough data in stream to read uint2')