from dummyserver.server import (
from dummyserver.testcase import SocketDummyServerTestCase, consume_socket
from urllib3 import HTTPConnectionPool, HTTPSConnectionPool, ProxyManager, util
from urllib3._collections import HTTPHeaderDict
from urllib3.connection import HTTPConnection, _get_default_user_agent
from urllib3.exceptions import (
from urllib3.packages.six.moves import http_client as httplib
from urllib3.poolmanager import proxy_from_url
from urllib3.util import ssl_, ssl_wrap_socket
from urllib3.util.retry import Retry
from urllib3.util.timeout import Timeout
from .. import LogRecorder, has_alpn, onlyPy3
import os
import os.path
import select
import shutil
import socket
import ssl
import sys
import tempfile
from collections import OrderedDict
from test import (
from threading import Event
import mock
import pytest
import trustme
def test_ua_header_can_be_overridden(self):
    headers = {'uSeR-AgENt': 'Definitely not urllib3!'}
    self.start_parsing_handler()
    expected_headers = {'Accept-Encoding': 'identity', 'Host': '{0}:{1}'.format(self.host, self.port)}
    expected_headers.update(headers)
    with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
        pool.request('GET', '/', headers=HTTPHeaderDict(headers))
        assert expected_headers == self.parsed_headers