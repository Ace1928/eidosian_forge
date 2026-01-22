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
def test_prepared_from_session(self, httpbin):

    class DummyAuth(requests.auth.AuthBase):

        def __call__(self, r):
            r.headers['Dummy-Auth-Test'] = 'dummy-auth-test-ok'
            return r
    req = requests.Request('GET', httpbin('headers'))
    assert not req.auth
    s = requests.Session()
    s.auth = DummyAuth()
    prep = s.prepare_request(req)
    resp = s.send(prep)
    assert resp.json()['headers']['Dummy-Auth-Test'] == 'dummy-auth-test-ok'