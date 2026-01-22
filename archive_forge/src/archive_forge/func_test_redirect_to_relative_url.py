import json
from test import LONG_TIMEOUT
import pytest
from dummyserver.server import HAS_IPV6
from dummyserver.testcase import HTTPDummyServerTestCase, IPv6HTTPDummyServerTestCase
from urllib3.connectionpool import port_by_scheme
from urllib3.exceptions import MaxRetryError, URLSchemeUnknown
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry
def test_redirect_to_relative_url(self):
    with PoolManager() as http:
        r = http.request('GET', '%s/redirect' % self.base_url, fields={'target': '/redirect'}, redirect=False)
        assert r.status == 303
        r = http.request('GET', '%s/redirect' % self.base_url, fields={'target': '/redirect'})
        assert r.status == 200
        assert r.data == b'Dummy server!'