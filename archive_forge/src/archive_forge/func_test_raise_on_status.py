import json
from test import LONG_TIMEOUT
import pytest
from dummyserver.server import HAS_IPV6
from dummyserver.testcase import HTTPDummyServerTestCase, IPv6HTTPDummyServerTestCase
from urllib3.connectionpool import port_by_scheme
from urllib3.exceptions import MaxRetryError, URLSchemeUnknown
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry
def test_raise_on_status(self):
    with PoolManager() as http:
        with pytest.raises(MaxRetryError):
            r = http.request('GET', '%s/status' % self.base_url, fields={'status': '500 Internal Server Error'}, retries=Retry(total=1, status_forcelist=range(500, 600)))
        with pytest.raises(MaxRetryError):
            r = http.request('GET', '%s/status' % self.base_url, fields={'status': '500 Internal Server Error'}, retries=Retry(total=1, status_forcelist=range(500, 600), raise_on_status=True))
        r = http.request('GET', '%s/status' % self.base_url, fields={'status': '500 Internal Server Error'}, retries=Retry(total=1, status_forcelist=range(500, 600), raise_on_status=False))
        assert r.status == 500