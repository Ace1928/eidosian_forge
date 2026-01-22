import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_string1(f):
    """
    >>> import io
    >>> read_string1(io.BytesIO(b"\\x00"))
    ''
    >>> read_string1(io.BytesIO(b"\\x03abcdef"))
    'abc'
    """
    n = read_uint1(f)
    assert n >= 0
    data = f.read(n)
    if len(data) == n:
        return data.decode('latin-1')
    raise ValueError('expected %d bytes in a string1, but only %d remain' % (n, len(data)))