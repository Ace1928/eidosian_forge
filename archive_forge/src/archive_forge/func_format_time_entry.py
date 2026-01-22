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
def format_time_entry(person, time, timezone_info):
    """Format an event."""
    timezone, timezone_neg_utc = timezone_info
    return b' '.join([person, str(time).encode('ascii'), format_timezone(timezone, timezone_neg_utc)])