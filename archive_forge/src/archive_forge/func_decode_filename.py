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
def decode_filename(filename):
    if isinstance(filename, bytes):
        try:
            filename_encoding = sys.getfilesystemencoding()
            if filename_encoding is None:
                filename_encoding = sys.getdefaultencoding()
            filename = filename.decode(filename_encoding)
        except UnicodeDecodeError:
            pass
    return filename