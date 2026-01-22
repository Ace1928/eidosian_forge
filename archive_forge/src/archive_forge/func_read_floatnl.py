import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_floatnl(f):
    """
    >>> import io
    >>> read_floatnl(io.BytesIO(b"-1.25\\n6"))
    -1.25
    """
    s = read_stringnl(f, decode=False, stripquotes=False)
    return float(s)