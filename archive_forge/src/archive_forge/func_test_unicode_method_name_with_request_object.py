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
def test_unicode_method_name_with_request_object(self, httpbin):
    files = {'file': open(__file__, 'rb')}
    s = requests.Session()
    req = requests.Request(u('POST'), httpbin('post'), files=files)
    prep = s.prepare_request(req)
    assert isinstance(prep.method, builtin_str)
    assert prep.method == 'POST'
    resp = s.send(prep)
    assert resp.status_code == 200