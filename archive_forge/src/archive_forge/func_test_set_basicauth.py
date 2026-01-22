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
@pytest.mark.parametrize('username, password', (('user', 'pass'), (u'имя'.encode('utf-8'), u'пароль'.encode('utf-8')), (42, 42), (None, None)))
def test_set_basicauth(self, httpbin, username, password):
    auth = (username, password)
    url = httpbin('get')
    r = requests.Request('GET', url, auth=auth)
    p = r.prepare()
    assert p.headers['Authorization'] == _basic_auth_str(username, password)