import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_unicodestring1(f):
    """
    >>> import io
    >>> s = 'abcd\\uabcd'
    >>> enc = s.encode('utf-8')
    >>> enc
    b'abcd\\xea\\xaf\\x8d'
    >>> n = bytes([len(enc)])  # little-endian 1-byte length
    >>> t = read_unicodestring1(io.BytesIO(n + enc + b'junk'))
    >>> s == t
    True

    >>> read_unicodestring1(io.BytesIO(n + enc[:-1]))
    Traceback (most recent call last):
    ...
    ValueError: expected 7 bytes in a unicodestring1, but only 6 remain
    """
    n = read_uint1(f)
    assert n >= 0
    data = f.read(n)
    if len(data) == n:
        return str(data, 'utf-8', 'surrogatepass')
    raise ValueError('expected %d bytes in a unicodestring1, but only %d remain' % (n, len(data)))