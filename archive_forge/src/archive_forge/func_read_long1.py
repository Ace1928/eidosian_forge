import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_long1(f):
    """
    >>> import io
    >>> read_long1(io.BytesIO(b"\\x00"))
    0
    >>> read_long1(io.BytesIO(b"\\x02\\xff\\x00"))
    255
    >>> read_long1(io.BytesIO(b"\\x02\\xff\\x7f"))
    32767
    >>> read_long1(io.BytesIO(b"\\x02\\x00\\xff"))
    -256
    >>> read_long1(io.BytesIO(b"\\x02\\x00\\x80"))
    -32768
    """
    n = read_uint1(f)
    data = f.read(n)
    if len(data) != n:
        raise ValueError('not enough data in stream to read long1')
    return decode_long(data)