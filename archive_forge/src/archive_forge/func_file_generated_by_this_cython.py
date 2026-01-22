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
def file_generated_by_this_cython(path):
    file_content = b''
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                file_content = f.read(len(GENERATED_BY_MARKER_BYTES))
        except (OSError, IOError):
            pass
    return file_content and file_content.startswith(GENERATED_BY_MARKER_BYTES)