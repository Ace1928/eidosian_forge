import binascii
import os
import posixpath
import stat
import warnings
import zlib
from collections import namedtuple
from hashlib import sha1
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
def format_timezone(offset, unnecessary_negative_timezone=False):
    """Format a timezone for Git serialization.

    Args:
      offset: Timezone offset as seconds difference to UTC
      unnecessary_negative_timezone: Whether to use a minus sign for
        UTC or positive timezones (-0000 and --700 rather than +0000 / +0700).
    """
    if offset % 60 != 0:
        raise ValueError('Unable to handle non-minute offset.')
    if offset < 0 or unnecessary_negative_timezone:
        sign = '-'
        offset = -offset
    else:
        sign = '+'
    return ('%c%02d%02d' % (sign, offset / 3600, offset / 60 % 60)).encode('ascii')