import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_uint1(f):
    """
    >>> import io
    >>> read_uint1(io.BytesIO(b'\\xff'))
    255
    """
    data = f.read(1)
    if data:
        return data[0]
    raise ValueError('not enough data in stream to read uint1')