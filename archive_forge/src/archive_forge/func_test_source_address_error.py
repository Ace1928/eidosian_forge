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
@pytest.mark.skipif(six.PY2 and platform.system() == 'Darwin' and (os.environ.get('GITHUB_ACTIONS') == 'true'), reason='fails on macOS 2.7 in GitHub Actions for an unknown reason')
def test_source_address_error(self):
    for addr in INVALID_SOURCE_ADDRESSES:
        with HTTPConnectionPool(self.host, self.port, source_address=addr, retries=False) as pool:
            with pytest.raises(NewConnectionError):
                pool.request('GET', '/source_address?{0}'.format(addr))