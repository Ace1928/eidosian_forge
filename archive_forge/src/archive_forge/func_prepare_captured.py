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
def prepare_captured(captured):
    captured_bytes = captured.strip()
    if not captured_bytes:
        return None
    for encoding in get_encoding_candidates():
        try:
            return captured_bytes.decode(encoding)
        except UnicodeDecodeError:
            pass
    return captured_bytes.decode('latin-1')