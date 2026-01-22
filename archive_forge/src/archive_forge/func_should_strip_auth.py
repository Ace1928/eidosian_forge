import os
import sys
import time
from datetime import timedelta
from collections import OrderedDict
from .auth import _basic_auth_str
from .compat import cookielib, is_py3, urljoin, urlparse, Mapping
from .cookies import (
from .models import Request, PreparedRequest, DEFAULT_REDIRECT_LIMIT
from .hooks import default_hooks, dispatch_hook
from ._internal_utils import to_native_string
from .utils import to_key_val_list, default_headers, DEFAULT_PORTS
from .exceptions import (
from .structures import CaseInsensitiveDict
from .adapters import HTTPAdapter
from .utils import (
from .status_codes import codes
from .models import REDIRECT_STATI
def should_strip_auth(self, old_url, new_url):
    """Decide whether Authorization header should be removed when redirecting"""
    old_parsed = urlparse(old_url)
    new_parsed = urlparse(new_url)
    if old_parsed.hostname != new_parsed.hostname:
        return True
    if old_parsed.scheme == 'http' and old_parsed.port in (80, None) and (new_parsed.scheme == 'https') and (new_parsed.port in (443, None)):
        return False
    changed_port = old_parsed.port != new_parsed.port
    changed_scheme = old_parsed.scheme != new_parsed.scheme
    default_port = (DEFAULT_PORTS.get(old_parsed.scheme, None), None)
    if not changed_scheme and old_parsed.port in default_port and (new_parsed.port in default_port):
        return False
    return changed_port or changed_scheme