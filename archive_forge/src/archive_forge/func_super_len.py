import codecs
import contextlib
import io
import os
import re
import socket
import struct
import sys
import tempfile
import warnings
import zipfile
from collections import OrderedDict
from urllib3.util import make_headers
from urllib3.util import parse_url
from .__version__ import __version__
from . import certs
from ._internal_utils import to_native_string
from .compat import parse_http_list as _parse_list_header
from .compat import (
from .cookies import cookiejar_from_dict
from .structures import CaseInsensitiveDict
from .exceptions import (
def super_len(o):
    total_length = None
    current_position = 0
    if hasattr(o, '__len__'):
        total_length = len(o)
    elif hasattr(o, 'len'):
        total_length = o.len
    elif hasattr(o, 'fileno'):
        try:
            fileno = o.fileno()
        except (io.UnsupportedOperation, AttributeError):
            pass
        else:
            total_length = os.fstat(fileno).st_size
            if 'b' not in o.mode:
                warnings.warn("Requests has determined the content-length for this request using the binary size of the file: however, the file has been opened in text mode (i.e. without the 'b' flag in the mode). This may lead to an incorrect content-length. In Requests 3.0, support will be removed for files in text mode.", FileModeWarning)
    if hasattr(o, 'tell'):
        try:
            current_position = o.tell()
        except (OSError, IOError):
            if total_length is not None:
                current_position = total_length
        else:
            if hasattr(o, 'seek') and total_length is None:
                try:
                    o.seek(0, 2)
                    total_length = o.tell()
                    o.seek(current_position or 0)
                except (OSError, IOError):
                    total_length = 0
    if total_length is None:
        total_length = 0
    return max(0, total_length - current_position)