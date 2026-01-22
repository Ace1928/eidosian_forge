from __future__ import absolute_import, division, print_function
from passlib.utils.compat import PY3
import base64
import calendar
import json
import logging; log = logging.getLogger(__name__)
import math
import struct
import sys
import time as _time
import re
from warnings import warn
from passlib import exc
from passlib.exc import TokenError, MalformedTokenError, InvalidTokenError, UsedTokenError
from passlib.utils import (to_unicode, to_bytes, consteq,
from passlib.utils.binary import BASE64_CHARS, b32encode, b32decode
from passlib.utils.compat import (u, unicode, native_string_types, bascii_to_str, int_types, num_types,
from passlib.utils.decor import hybrid_method, memoized_property
from passlib.crypto.digest import lookup_hash, compile_hmac, pbkdf2_hmac
from passlib.hash import pbkdf2_sha256
@hybrid_method
def normalize_token(self_or_cls, token):
    """
        Normalize OTP token representation:
        strips whitespace, converts integers to a zero-padded string,
        validates token content & number of digits.

        This is a hybrid method -- it can be called at the class level,
        as ``TOTP.normalize_token()``, or the instance level as ``TOTP().normalize_token()``.
        It will normalize to the instance-specific number of :attr:`~TOTP.digits`,
        or use the class default.

        :arg token:
            token as ascii bytes, unicode, or an integer.

        :raises ValueError:
            if token has wrong number of digits, or contains non-numeric characters.

        :returns:
            token as :class:`!unicode` string, containing only digits 0-9.
        """
    digits = self_or_cls.digits
    if isinstance(token, int_types):
        token = u('%0*d') % (digits, token)
    else:
        token = to_unicode(token, param='token')
        token = _clean_re.sub(u(''), token)
        if not token.isdigit():
            raise MalformedTokenError('Token must contain only the digits 0-9')
    if len(token) != digits:
        raise MalformedTokenError('Token must have exactly %d digits' % digits)
    return token