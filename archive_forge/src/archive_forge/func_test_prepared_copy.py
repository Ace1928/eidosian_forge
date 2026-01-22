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
@pytest.mark.parametrize('kwargs', (None, {'method': 'GET', 'url': 'http://www.example.com', 'data': 'foo=bar', 'hooks': default_hooks()}, {'method': 'GET', 'url': 'http://www.example.com', 'data': 'foo=bar', 'hooks': default_hooks(), 'cookies': {'foo': 'bar'}}, {'method': 'GET', 'url': u('http://www.example.com/üniçø∂é')}))
def test_prepared_copy(kwargs):
    p = PreparedRequest()
    if kwargs:
        p.prepare(**kwargs)
    copy = p.copy()
    for attr in ('method', 'url', 'headers', '_cookies', 'body', 'hooks'):
        assert getattr(p, attr) == getattr(copy, attr)