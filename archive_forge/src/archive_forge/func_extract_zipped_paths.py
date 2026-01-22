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
def extract_zipped_paths(path):
    """Replace nonexistent paths that look like they refer to a member of a zip
    archive with the location of an extracted copy of the target, or else
    just return the provided path unchanged.
    """
    if os.path.exists(path):
        return path
    archive, member = os.path.split(path)
    while archive and (not os.path.exists(archive)):
        archive, prefix = os.path.split(archive)
        if not prefix:
            break
        member = '/'.join([prefix, member])
    if not zipfile.is_zipfile(archive):
        return path
    zip_file = zipfile.ZipFile(archive)
    if member not in zip_file.namelist():
        return path
    tmp = tempfile.gettempdir()
    extracted_path = os.path.join(tmp, member.split('/')[-1])
    if not os.path.exists(extracted_path):
        with atomic_open(extracted_path) as file_handler:
            file_handler.write(zip_file.read(member))
    return extracted_path