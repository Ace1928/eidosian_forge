from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def b64s_decode(data):
    """
    decode from shortened base64 format which omits padding & whitespace.
    uses default ``+/`` altchars.
    """
    if isinstance(data, unicode):
        try:
            data = data.encode('ascii')
        except UnicodeEncodeError:
            raise suppress_cause(ValueError('string argument should contain only ASCII characters'))
    off = len(data) & 3
    if off == 0:
        pass
    elif off == 2:
        data += _BASE64_PAD2
    elif off == 3:
        data += _BASE64_PAD1
    else:
        raise ValueError('invalid base64 input')
    try:
        return a2b_base64(data)
    except _BinAsciiError as err:
        raise suppress_cause(TypeError(err))