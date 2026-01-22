from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def encode_transposed_bytes(self, source, offsets):
    """encode byte string, first transposing source using offset list"""
    if not isinstance(source, bytes):
        raise TypeError('source must be bytes, not %s' % (type(source),))
    tmp = join_byte_elems((source[off] for off in offsets))
    return self.encode_bytes(tmp)