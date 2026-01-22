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
def test_prepared_request_with_file_is_pickleable(self, httpbin):
    files = {'file': open(__file__, 'rb')}
    r = requests.Request('POST', httpbin('post'), files=files)
    p = r.prepare()
    r = pickle.loads(pickle.dumps(p))
    assert r.url == p.url
    assert r.headers == p.headers
    assert r.body == p.body
    s = requests.Session()
    resp = s.send(r)
    assert resp.status_code == 200