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
def test_DIGEST_HTTP_200_OK_GET(self, httpbin):
    for authtype in self.digest_auth_algo:
        auth = HTTPDigestAuth('user', 'pass')
        url = httpbin('digest-auth', 'auth', 'user', 'pass', authtype, 'never')
        r = requests.get(url, auth=auth)
        assert r.status_code == 200
        r = requests.get(url)
        assert r.status_code == 401
        print(r.headers['WWW-Authenticate'])
        s = requests.session()
        s.auth = HTTPDigestAuth('user', 'pass')
        r = s.get(url)
        assert r.status_code == 200