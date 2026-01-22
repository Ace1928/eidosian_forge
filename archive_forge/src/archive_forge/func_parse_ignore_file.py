import errno
import os
from io import BytesIO
from typing import Set
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import bedding
def parse_ignore_file(f):
    """Read in all of the lines in the file and turn it into an ignore list

    Continue in the case of utf8 decoding errors, and emit a warning when
    such and error is found. Optimise for the common case -- no decoding
    errors.
    """
    ignored = set()
    ignore_file = f.read()
    try:
        unicode_lines = ignore_file.decode('utf8').split('\n')
    except UnicodeDecodeError:
        lines = ignore_file.split(b'\n')
        unicode_lines = []
        for line_number, line in enumerate(lines):
            try:
                unicode_lines.append(line.decode('utf-8'))
            except UnicodeDecodeError:
                trace.warning('.bzrignore: On Line #%d, malformed utf8 character. Ignoring line.' % (line_number + 1))
    for line in unicode_lines:
        line = line.rstrip('\r\n')
        if not line or line.startswith('#'):
            continue
        ignored.add(globbing.normalize_pattern(line))
    return ignored