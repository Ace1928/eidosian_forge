import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def make_compat_str(o: object) -> str:
    """Converts everything to string, if bytes guessing the encoding."""
    if isinstance(o, bytes):
        enc = charset_normalizer.detect(o)
        try:
            return o.decode(enc['encoding'])
        except UnicodeDecodeError:
            return str(o)
    else:
        return str(o)