from __future__ import division
import re
import stat
from .helpers import (
def format_who_when(fields):
    """Format tuple of name,email,secs-since-epoch,utc-offset-secs as bytes."""
    offset = fields[3]
    if offset < 0:
        offset_sign = b'-'
        offset = abs(offset)
    else:
        offset_sign = b'+'
    offset_hours = offset // 3600
    offset_minutes = offset // 60 - offset_hours * 60
    offset_str = offset_sign + ('%02d%02d' % (offset_hours, offset_minutes)).encode('ascii')
    name = fields[0]
    if name == b'':
        sep = b''
    else:
        sep = b' '
    name = utf8_bytes_string(name)
    email = fields[1]
    email = utf8_bytes_string(email)
    return b''.join((name, sep, b'<', email, b'> ', ('%d' % fields[2]).encode('ascii'), b' ', offset_str))