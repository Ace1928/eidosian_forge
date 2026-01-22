import json
from test import LONG_TIMEOUT
import pytest
from dummyserver.server import HAS_IPV6
from dummyserver.testcase import HTTPDummyServerTestCase, IPv6HTTPDummyServerTestCase
from urllib3.connectionpool import port_by_scheme
from urllib3.exceptions import MaxRetryError, URLSchemeUnknown
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry
def test_http_with_ssl_keywords(self):
    with PoolManager(ca_certs='REQUIRED') as http:
        r = http.request('GET', 'http://%s:%s/' % (self.host, self.port))
        assert r.status == 200