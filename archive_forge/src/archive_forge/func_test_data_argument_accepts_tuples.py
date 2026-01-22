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
@pytest.mark.parametrize('data', ((('a', 'b'), ('c', 'd')), (('c', 'd'), ('a', 'b')), (('a', 'b'), ('c', 'd'), ('e', 'f'))))
def test_data_argument_accepts_tuples(data):
    """Ensure that the data argument will accept tuples of strings
    and properly encode them.
    """
    p = PreparedRequest()
    p.prepare(method='GET', url='http://www.example.com', data=data, hooks=default_hooks())
    assert p.body == urlencode(data)