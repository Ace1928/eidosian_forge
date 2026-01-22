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
def proxy_bypass_registry(host):
    try:
        if is_py3:
            import winreg
        else:
            import _winreg as winreg
    except ImportError:
        return False
    try:
        internetSettings = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings')
        proxyEnable = int(winreg.QueryValueEx(internetSettings, 'ProxyEnable')[0])
        proxyOverride = winreg.QueryValueEx(internetSettings, 'ProxyOverride')[0]
    except OSError:
        return False
    if not proxyEnable or not proxyOverride:
        return False
    proxyOverride = proxyOverride.split(';')
    for test in proxyOverride:
        if test == '<local>':
            if '.' not in host:
                return True
        test = test.replace('.', '\\.')
        test = test.replace('*', '.*')
        test = test.replace('?', '.')
        if re.match(test, host, re.I):
            return True
    return False