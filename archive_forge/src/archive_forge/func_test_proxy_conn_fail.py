import json
import os.path
import shutil
import socket
import ssl
import sys
import tempfile
import warnings
from test import (
import pytest
import trustme
from dummyserver.server import DEFAULT_CA, HAS_IPV6, get_unreachable_address
from dummyserver.testcase import HTTPDummyProxyTestCase, IPv6HTTPDummyProxyTestCase
from urllib3._collections import HTTPHeaderDict
from urllib3.connectionpool import VerifiedHTTPSConnection, connection_from_url
from urllib3.exceptions import (
from urllib3.poolmanager import ProxyManager, proxy_from_url
from urllib3.util import Timeout
from urllib3.util.ssl_ import create_urllib3_context
from .. import TARPIT_HOST, requires_network
def test_proxy_conn_fail(self):
    host, port = get_unreachable_address()
    with proxy_from_url('http://%s:%s/' % (host, port), retries=1, timeout=LONG_TIMEOUT) as http:
        with pytest.raises(MaxRetryError):
            http.request('GET', '%s/' % self.https_url)
        with pytest.raises(MaxRetryError):
            http.request('GET', '%s/' % self.http_url)
        with pytest.raises(MaxRetryError) as e:
            http.request('GET', '%s/' % self.http_url)
        assert type(e.value.reason) == ProxyError