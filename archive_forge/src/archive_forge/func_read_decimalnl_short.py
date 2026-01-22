import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_decimalnl_short(f):
    """
    >>> import io
    >>> read_decimalnl_short(io.BytesIO(b"1234\\n56"))
    1234

    >>> read_decimalnl_short(io.BytesIO(b"1234L\\n56"))
    Traceback (most recent call last):
    ...
    ValueError: invalid literal for int() with base 10: b'1234L'
    """
    s = read_stringnl(f, decode=False, stripquotes=False)
    if s == b'00':
        return False
    elif s == b'01':
        return True
    return int(s)