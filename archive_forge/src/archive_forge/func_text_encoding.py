import os
import abc
import codecs
import errno
import stat
import sys
from _thread import allocate_lock as Lock
import io
from io import (__all__, SEEK_SET, SEEK_CUR, SEEK_END)
from _io import FileIO
def text_encoding(encoding, stacklevel=2):
    """
    A helper function to choose the text encoding.

    When encoding is not None, this function returns it.
    Otherwise, this function returns the default text encoding
    (i.e. "locale" or "utf-8" depends on UTF-8 mode).

    This function emits an EncodingWarning if *encoding* is None and
    sys.flags.warn_default_encoding is true.

    This can be used in APIs with an encoding=None parameter
    that pass it to TextIOWrapper or open.
    However, please consider using encoding="utf-8" for new APIs.
    """
    if encoding is None:
        if sys.flags.utf8_mode:
            encoding = 'utf-8'
        else:
            encoding = 'locale'
        if sys.flags.warn_default_encoding:
            import warnings
            warnings.warn("'encoding' argument not specified.", EncodingWarning, stacklevel + 1)
    return encoding