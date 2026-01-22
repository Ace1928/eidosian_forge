from __future__ import absolute_import
import re
import sys
import struct
from .compat import PY3, unichr
from .scanner import make_scanner, JSONDecodeError
def scan_four_digit_hex(s, end, _m=re.compile('^[0-9a-fA-F]{4}$').match):
    """Scan a four digit hex number from s[end:end + 4]
    """
    msg = 'Invalid \\uXXXX escape sequence'
    esc = s[end:end + 4]
    if not _m(esc):
        raise JSONDecodeError(msg, s, end - 2)
    try:
        return (int(esc, 16), end + 4)
    except ValueError:
        raise JSONDecodeError(msg, s, end - 2)