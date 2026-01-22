import json
from test import LONG_TIMEOUT
import pytest
from dummyserver.server import HAS_IPV6
from dummyserver.testcase import HTTPDummyServerTestCase, IPv6HTTPDummyServerTestCase
from urllib3.connectionpool import port_by_scheme
from urllib3.exceptions import MaxRetryError, URLSchemeUnknown
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry
def test_redirect_cross_host_set_removed_headers(self):
    with PoolManager() as http:
        r = http.request('GET', '%s/redirect' % self.base_url, fields={'target': '%s/headers' % self.base_url_alt}, headers={'X-API-Secret': 'foo', 'Authorization': 'bar'}, retries=Retry(remove_headers_on_redirect=['X-API-Secret']))
        assert r.status == 200
        data = json.loads(r.data.decode('utf-8'))
        assert 'X-API-Secret' not in data
        assert data['Authorization'] == 'bar'
        r = http.request('GET', '%s/redirect' % self.base_url, fields={'target': '%s/headers' % self.base_url_alt}, headers={'x-api-secret': 'foo', 'authorization': 'bar'}, retries=Retry(remove_headers_on_redirect=['X-API-Secret']))
        assert r.status == 200
        data = json.loads(r.data.decode('utf-8'))
        assert 'x-api-secret' not in data
        assert 'X-API-Secret' not in data
        assert data['Authorization'] == 'bar'