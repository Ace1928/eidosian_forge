import json
from test import LONG_TIMEOUT
import pytest
from dummyserver.server import HAS_IPV6
from dummyserver.testcase import HTTPDummyServerTestCase, IPv6HTTPDummyServerTestCase
from urllib3.connectionpool import port_by_scheme
from urllib3.exceptions import MaxRetryError, URLSchemeUnknown
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry
def test_redirect_without_preload_releases_connection(self):
    with PoolManager(block=True, maxsize=2) as http:
        r = http.request('GET', '%s/redirect' % self.base_url, preload_content=False)
        assert r._pool.num_requests == 2
        assert r._pool.num_connections == 1
        assert len(http.pools) == 1