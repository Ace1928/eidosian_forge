import io
import json
import logging
import os
import platform
import socket
import sys
import time
import warnings
from test import LONG_TIMEOUT, SHORT_TIMEOUT, onlyPy2
from threading import Event
import mock
import pytest
import six
from dummyserver.server import HAS_IPV6_AND_DNS, NoIPv6Warning
from dummyserver.testcase import HTTPDummyServerTestCase, SocketDummyServerTestCase
from urllib3 import HTTPConnectionPool, encode_multipart_formdata
from urllib3._collections import HTTPHeaderDict
from urllib3.connection import _get_default_user_agent
from urllib3.exceptions import (
from urllib3.packages.six import b, u
from urllib3.packages.six.moves.urllib.parse import urlencode
from urllib3.util import SKIP_HEADER, SKIPPABLE_HEADERS
from urllib3.util.retry import RequestHistory, Retry
from urllib3.util.timeout import Timeout
from .. import INVALID_SOURCE_ADDRESSES, TARPIT_HOST, VALID_SOURCE_ADDRESSES
from ..port_helpers import find_unused_port
@pytest.mark.parametrize('user_agent', [u'Schönefeld/1.18.0', u'Schönefeld/1.18.0'.encode('iso-8859-1')])
def test_user_agent_non_ascii_user_agent(self, user_agent):
    if six.PY2 and (not isinstance(user_agent, str)):
        pytest.skip('Python 2 raises UnicodeEncodeError when passed a unicode header')
    with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
        r = pool.urlopen('GET', '/headers', headers={'User-Agent': user_agent})
        request_headers = json.loads(r.data.decode('utf8'))
        assert 'User-Agent' in request_headers
        assert request_headers['User-Agent'] == u'Schönefeld/1.18.0'