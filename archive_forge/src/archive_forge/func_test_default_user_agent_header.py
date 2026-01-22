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
def test_default_user_agent_header(self):
    """ConnectionPool has a default user agent"""
    default_ua = _get_default_user_agent()
    custom_ua = "I'm not a web scraper, what are you talking about?"
    custom_ua2 = 'Yet Another User Agent'
    with HTTPConnectionPool(self.host, self.port) as pool:
        r = pool.request('GET', '/headers')
        request_headers = json.loads(r.data.decode('utf8'))
        assert request_headers.get('User-Agent') == _get_default_user_agent()
        headers = {'UsEr-AGENt': custom_ua}
        r = pool.request('GET', '/headers', headers=headers)
        request_headers = json.loads(r.data.decode('utf8'))
        assert request_headers.get('User-Agent') == custom_ua
        pool_headers = {'foo': 'bar'}
        pool.headers = pool_headers
        r = pool.request('GET', '/headers')
        request_headers = json.loads(r.data.decode('utf8'))
        assert request_headers.get('User-Agent') == default_ua
        assert 'User-Agent' not in pool_headers
        pool.headers.update({'User-Agent': custom_ua2})
        r = pool.request('GET', '/headers')
        request_headers = json.loads(r.data.decode('utf8'))
        assert request_headers.get('User-Agent') == custom_ua2