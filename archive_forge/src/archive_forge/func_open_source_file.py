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
def open_source_file(source_filename, encoding=None, error_handling=None):
    stream = None
    try:
        if encoding is None:
            f = io.open(source_filename, 'rb')
            encoding = detect_opened_file_encoding(f)
            f.seek(0)
            stream = io.TextIOWrapper(f, encoding=encoding, errors=error_handling)
        else:
            stream = io.open(source_filename, encoding=encoding, errors=error_handling)
    except OSError:
        if os.path.exists(source_filename):
            raise
        try:
            loader = __loader__
            if source_filename.startswith(loader.archive):
                stream = open_source_from_loader(loader, source_filename, encoding, error_handling)
        except (NameError, AttributeError):
            pass
    if stream is None:
        raise FileNotFoundError(source_filename)
    skip_bom(stream)
    return stream