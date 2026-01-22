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
def test_response_decode_unicode(self):
    """When called with decode_unicode, Response.iter_content should always
        return unicode.
        """
    r = requests.Response()
    r._content_consumed = True
    r._content = b'the content'
    r.encoding = 'ascii'
    chunks = r.iter_content(decode_unicode=True)
    assert all((isinstance(chunk, str) for chunk in chunks))
    r = requests.Response()
    r.raw = io.BytesIO(b'the content')
    r.encoding = 'ascii'
    chunks = r.iter_content(decode_unicode=True)
    assert all((isinstance(chunk, str) for chunk in chunks))