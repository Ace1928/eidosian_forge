import errno
import os
import stat
import sys
from subprocess import check_output
def ensure_byte_string(value):
    """
    Return the given ``value`` as bytestring.

    If the given ``value`` is not a byte string, but a real unicode string, it
    is encoded with the filesystem encoding (as in
    :func:`sys.getfilesystemencoding()`).
    """
    if not isinstance(value, bytes):
        value = value.encode(sys.getfilesystemencoding())
    return value