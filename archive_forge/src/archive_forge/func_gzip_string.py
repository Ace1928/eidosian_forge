import errno
import os
from io import BytesIO
from .lazy_import import lazy_import
import gzip
import itertools
import patiencediff
from breezy import (
from . import errors
from .i18n import gettext
def gzip_string(lines):
    sio = BytesIO()
    with gzip.GzipFile(None, mode='wb', fileobj=sio) as data_file:
        data_file.writelines(lines)
    return sio.getvalue()