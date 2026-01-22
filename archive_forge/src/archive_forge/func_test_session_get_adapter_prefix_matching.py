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
def test_session_get_adapter_prefix_matching(self):
    prefix = 'https://example.com'
    more_specific_prefix = prefix + '/some/path'
    url_matching_only_prefix = prefix + '/another/path'
    url_matching_more_specific_prefix = more_specific_prefix + '/longer/path'
    url_not_matching_prefix = 'https://another.example.com/'
    s = requests.Session()
    prefix_adapter = HTTPAdapter()
    more_specific_prefix_adapter = HTTPAdapter()
    s.mount(prefix, prefix_adapter)
    s.mount(more_specific_prefix, more_specific_prefix_adapter)
    assert s.get_adapter(url_matching_only_prefix) is prefix_adapter
    assert s.get_adapter(url_matching_more_specific_prefix) is more_specific_prefix_adapter
    assert s.get_adapter(url_not_matching_prefix) not in (prefix_adapter, more_specific_prefix_adapter)