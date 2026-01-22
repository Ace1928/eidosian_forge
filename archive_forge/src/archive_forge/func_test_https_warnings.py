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
def test_https_warnings(self, nosan_server):
    """warnings are emitted with requests.get"""
    host, port, ca_bundle = nosan_server
    if HAS_MODERN_SSL or HAS_PYOPENSSL:
        warnings_expected = ('SubjectAltNameWarning',)
    else:
        warnings_expected = ('SNIMissingWarning', 'InsecurePlatformWarning', 'SubjectAltNameWarning')
    with pytest.warns(None) as warning_records:
        warnings.simplefilter('always')
        requests.get('https://localhost:{}/'.format(port), verify=ca_bundle)
    warning_records = [item for item in warning_records if item.category.__name__ != 'ResourceWarning']
    warnings_category = tuple((item.category.__name__ for item in warning_records))
    assert warnings_category == warnings_expected