from __future__ import division
import json
import os
import pickle
import collections
import contextlib
import warnings
import re
import io
import requests
import pytest
import urllib3
from requests.adapters import HTTPAdapter
from requests.auth import HTTPDigestAuth, _basic_auth_str
from requests.compat import (
from requests.cookies import (
from requests.exceptions import (
from requests.exceptions import SSLError as RequestsSSLError
from requests.models import PreparedRequest
from requests.structures import CaseInsensitiveDict
from requests.sessions import SessionRedirectMixin
from requests.models import urlencode
from requests.hooks import default_hooks
from requests.compat import JSONDecodeError, is_py3, MutableMapping
from .compat import StringIO, u
from .utils import override_environ
from urllib3.util import Timeout as Urllib3Timeout
def test_transport_adapter_ordering(self):
    s = requests.Session()
    order = ['https://', 'http://']
    assert order == list(s.adapters)
    s.mount('http://git', HTTPAdapter())
    s.mount('http://github', HTTPAdapter())
    s.mount('http://github.com', HTTPAdapter())
    s.mount('http://github.com/about/', HTTPAdapter())
    order = ['http://github.com/about/', 'http://github.com', 'http://github', 'http://git', 'https://', 'http://']
    assert order == list(s.adapters)
    s.mount('http://gittip', HTTPAdapter())
    s.mount('http://gittip.com', HTTPAdapter())
    s.mount('http://gittip.com/about/', HTTPAdapter())
    order = ['http://github.com/about/', 'http://gittip.com/about/', 'http://github.com', 'http://gittip.com', 'http://github', 'http://gittip', 'http://git', 'https://', 'http://']
    assert order == list(s.adapters)
    s2 = requests.Session()
    s2.adapters = {'http://': HTTPAdapter()}
    s2.mount('https://', HTTPAdapter())
    assert 'http://' in s2.adapters
    assert 'https://' in s2.adapters