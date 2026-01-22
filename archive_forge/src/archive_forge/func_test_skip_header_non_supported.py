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
@pytest.mark.parametrize('header', ['Content-Length', 'content-length'])
@pytest.mark.parametrize('chunked', [True, False])
def test_skip_header_non_supported(self, header, chunked):
    with HTTPConnectionPool(self.host, self.port) as pool:
        with pytest.raises(ValueError) as e:
            pool.request('GET', '/headers', headers={header: SKIP_HEADER}, chunked=chunked)
        assert str(e.value) == "urllib3.util.SKIP_HEADER only supports 'Accept-Encoding', 'Host', 'User-Agent'"
        assert all(("'" + header.title() + "'" in str(e.value) for header in SKIPPABLE_HEADERS))