from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def encode_int6(self, value):
    """encodes 6-bit integer -> single hash64 character"""
    if value < 0 or value > 63:
        raise ValueError('value out of range')
    if PY3:
        return self.bytemap[value:value + 1]
    else:
        return self._encode64(value)