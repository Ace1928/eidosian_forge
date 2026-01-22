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
def test_cross_host_redirect(self):
    with proxy_from_url(self.proxy_url) as http:
        cross_host_location = '%s/echo?a=b' % self.http_url_alt
        with pytest.raises(MaxRetryError):
            http.request('GET', '%s/redirect' % self.http_url, fields={'target': cross_host_location}, retries=0)
        r = http.request('GET', '%s/redirect' % self.http_url, fields={'target': '%s/echo?a=b' % self.http_url_alt}, retries=1)
        assert r._pool.host != self.http_host_alt