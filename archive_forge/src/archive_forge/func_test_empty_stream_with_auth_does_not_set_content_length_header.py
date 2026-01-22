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
def test_empty_stream_with_auth_does_not_set_content_length_header(self, httpbin):
    """Ensure that a byte stream with size 0 will not set both a Content-Length
        and Transfer-Encoding header.
        """
    auth = ('user', 'pass')
    url = httpbin('post')
    file_obj = io.BytesIO(b'')
    r = requests.Request('POST', url, auth=auth, data=file_obj)
    prepared_request = r.prepare()
    assert 'Transfer-Encoding' in prepared_request.headers
    assert 'Content-Length' not in prepared_request.headers