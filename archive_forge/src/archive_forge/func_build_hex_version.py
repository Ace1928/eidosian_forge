from __future__ import absolute_import
import cython
import os
import sys
import re
import io
import codecs
import glob
import shutil
import tempfile
from functools import wraps
from . import __version__ as cython_version
def build_hex_version(version_string):
    """
    Parse and translate public version identifier like '4.3a1' into the readable hex representation '0x040300A1' (like PY_VERSION_HEX).

    SEE: https://peps.python.org/pep-0440/#public-version-identifiers
    """
    digits = []
    release_status = 240
    for segment in re.split('(\\D+)', version_string):
        if segment in ('a', 'b', 'rc'):
            release_status = {'a': 160, 'b': 176, 'rc': 192}[segment]
            digits = (digits + [0, 0])[:3]
        elif segment in ('.dev', '.pre', '.post'):
            break
        elif segment != '.':
            digits.append(int(segment))
    digits = (digits + [0] * 3)[:4]
    digits[3] += release_status
    hexversion = 0
    for digit in digits:
        hexversion = (hexversion << 8) + digit
    return '0x%08X' % hexversion