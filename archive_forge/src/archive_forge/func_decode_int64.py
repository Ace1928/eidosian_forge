from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def decode_int64(self, source):
    """decode 11 char base64 string -> 64-bit integer

        this format is used primarily by des-crypt & variants to encode
        the DES output value used as a checksum.
        """
    return self._decode_int(source, 64)