import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_unicodestring8(f):
    """
    >>> import io
    >>> s = 'abcd\\uabcd'
    >>> enc = s.encode('utf-8')
    >>> enc
    b'abcd\\xea\\xaf\\x8d'
    >>> n = bytes([len(enc)]) + b'\\0' * 7  # little-endian 8-byte length
    >>> t = read_unicodestring8(io.BytesIO(n + enc + b'junk'))
    >>> s == t
    True

    >>> read_unicodestring8(io.BytesIO(n + enc[:-1]))
    Traceback (most recent call last):
    ...
    ValueError: expected 7 bytes in a unicodestring8, but only 6 remain
    """
    n = read_uint8(f)
    assert n >= 0
    if n > sys.maxsize:
        raise ValueError('unicodestring8 byte count > sys.maxsize: %d' % n)
    data = f.read(n)
    if len(data) == n:
        return str(data, 'utf-8', 'surrogatepass')
    raise ValueError('expected %d bytes in a unicodestring8, but only %d remain' % (n, len(data)))