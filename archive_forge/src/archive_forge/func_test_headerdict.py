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
def test_headerdict(self):
    default_headers = HTTPHeaderDict(a='b')
    proxy_headers = HTTPHeaderDict()
    proxy_headers.add('foo', 'bar')
    with proxy_from_url(self.proxy_url, headers=default_headers, proxy_headers=proxy_headers) as http:
        request_headers = HTTPHeaderDict(baz='quux')
        r = http.request('GET', '%s/headers' % self.http_url, headers=request_headers)
        returned_headers = json.loads(r.data.decode())
        assert returned_headers.get('Foo') == 'bar'
        assert returned_headers.get('Baz') == 'quux'