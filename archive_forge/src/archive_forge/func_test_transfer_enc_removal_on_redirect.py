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
def test_transfer_enc_removal_on_redirect(self, httpbin):
    purged_headers = ('Transfer-Encoding', 'Content-Type')
    ses = requests.Session()
    req = requests.Request('POST', httpbin('post'), data=(b'x' for x in range(1)))
    prep = ses.prepare_request(req)
    assert 'Transfer-Encoding' in prep.headers
    resp = requests.Response()
    resp.raw = io.BytesIO(b'the content')
    resp.request = prep
    setattr(resp.raw, 'release_conn', lambda *args: args)
    resp.status_code = 302
    resp.headers['location'] = httpbin('get')
    next_resp = next(ses.resolve_redirects(resp, prep))
    assert next_resp.request.body is None
    for header in purged_headers:
        assert header not in next_resp.request.headers