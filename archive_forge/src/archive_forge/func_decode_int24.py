from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def decode_int24(self, source):
    """decodes 4 char string -> 24-bit integer"""
    if not isinstance(source, bytes):
        raise TypeError('source must be bytes, not %s' % (type(source),))
    if len(source) != 4:
        raise ValueError('source must be exactly 4 bytes')
    decode = self._decode64
    try:
        if self.big:
            return decode(source[3]) + (decode(source[2]) << 6) + (decode(source[1]) << 12) + (decode(source[0]) << 18)
        else:
            return decode(source[0]) + (decode(source[1]) << 6) + (decode(source[2]) << 12) + (decode(source[3]) << 18)
    except KeyError:
        raise ValueError('invalid character')