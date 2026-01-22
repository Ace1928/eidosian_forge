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
def test_basicauth_with_netrc(self, httpbin):
    auth = ('user', 'pass')
    wrong_auth = ('wronguser', 'wrongpass')
    url = httpbin('basic-auth', 'user', 'pass')
    old_auth = requests.sessions.get_netrc_auth
    try:

        def get_netrc_auth_mock(url):
            return auth
        requests.sessions.get_netrc_auth = get_netrc_auth_mock
        r = requests.get(url)
        assert r.status_code == 200
        r = requests.get(url, auth=wrong_auth)
        assert r.status_code == 401
        s = requests.session()
        r = s.get(url)
        assert r.status_code == 200
        s.auth = wrong_auth
        r = s.get(url)
        assert r.status_code == 401
    finally:
        requests.sessions.get_netrc_auth = old_auth