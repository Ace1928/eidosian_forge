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
def get_netrc_auth(url, raise_errors=False):
    """Returns the Requests tuple auth for a given url from netrc."""
    netrc_file = os.environ.get('NETRC')
    if netrc_file is not None:
        netrc_locations = (netrc_file,)
    else:
        netrc_locations = ('~/{}'.format(f) for f in NETRC_FILES)
    try:
        from netrc import netrc, NetrcParseError
        netrc_path = None
        for f in netrc_locations:
            try:
                loc = os.path.expanduser(f)
            except KeyError:
                return
            if os.path.exists(loc):
                netrc_path = loc
                break
        if netrc_path is None:
            return
        ri = urlparse(url)
        splitstr = b':'
        if isinstance(url, str):
            splitstr = splitstr.decode('ascii')
        host = ri.netloc.split(splitstr)[0]
        try:
            _netrc = netrc(netrc_path).authenticators(host)
            if _netrc:
                login_i = 0 if _netrc[0] else 1
                return (_netrc[login_i], _netrc[2])
        except (NetrcParseError, IOError):
            if raise_errors:
                raise
    except (ImportError, AttributeError):
        pass