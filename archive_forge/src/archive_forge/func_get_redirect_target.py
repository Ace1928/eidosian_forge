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
def get_redirect_target(self, resp):
    """Receives a Response. Returns a redirect URI or ``None``"""
    if resp.is_redirect:
        location = resp.headers['location']
        if is_py3:
            location = location.encode('latin1')
        return to_native_string(location, 'utf8')
    return None