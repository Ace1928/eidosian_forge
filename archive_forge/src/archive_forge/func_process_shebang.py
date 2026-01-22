from __future__ import unicode_literals
import base64
import codecs
import datetime
from email import message_from_file
import hashlib
import json
import logging
import os
import posixpath
import re
import shutil
import sys
import tempfile
import zipfile
from . import __version__, DistlibException
from .compat import sysconfig, ZipFile, fsdecode, text_type, filter
from .database import InstalledDistribution
from .metadata import Metadata, WHEEL_METADATA_FILENAME, LEGACY_METADATA_FILENAME
from .util import (FileOperator, convert_path, CSVReader, CSVWriter, Cache,
from .version import NormalizedVersion, UnsupportedVersionError
def process_shebang(self, data):
    m = SHEBANG_RE.match(data)
    if m:
        end = m.end()
        shebang, data_after_shebang = (data[:end], data[end:])
        if b'pythonw' in shebang.lower():
            shebang_python = SHEBANG_PYTHONW
        else:
            shebang_python = SHEBANG_PYTHON
        m = SHEBANG_DETAIL_RE.match(shebang)
        if m:
            args = b' ' + m.groups()[-1]
        else:
            args = b''
        shebang = shebang_python + args
        data = shebang + data_after_shebang
    else:
        cr = data.find(b'\r')
        lf = data.find(b'\n')
        if cr < 0 or cr > lf:
            term = b'\n'
        elif data[cr:cr + 2] == b'\r\n':
            term = b'\r\n'
        else:
            term = b'\r'
        data = SHEBANG_PYTHON + term + data
    return data