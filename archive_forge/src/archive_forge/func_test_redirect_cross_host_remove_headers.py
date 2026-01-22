import json
from test import LONG_TIMEOUT
import pytest
from dummyserver.server import HAS_IPV6
from dummyserver.testcase import HTTPDummyServerTestCase, IPv6HTTPDummyServerTestCase
from urllib3.connectionpool import port_by_scheme
from urllib3.exceptions import MaxRetryError, URLSchemeUnknown
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry
def test_redirect_cross_host_remove_headers(self):
    with PoolManager() as http:
        r = http.request('GET', '%s/redirect' % self.base_url, fields={'target': '%s/headers' % self.base_url_alt}, headers={'Authorization': 'foo'})
        assert r.status == 200
        data = json.loads(r.data.decode('utf-8'))
        assert 'Authorization' not in data
        r = http.request('GET', '%s/redirect' % self.base_url, fields={'target': '%s/headers' % self.base_url_alt}, headers={'authorization': 'foo'})
        assert r.status == 200
        data = json.loads(r.data.decode('utf-8'))
        assert 'authorization' not in data
        assert 'Authorization' not in data