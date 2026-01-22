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
def test_HTTP_302_TOO_MANY_REDIRECTS(self, httpbin):
    try:
        requests.get(httpbin('relative-redirect', '50'))
    except TooManyRedirects as e:
        url = httpbin('relative-redirect', '20')
        assert e.request.url == url
        assert e.response.url == url
        assert len(e.response.history) == 30
    else:
        pytest.fail('Expected redirect to raise TooManyRedirects but it did not')