import json
from test import LONG_TIMEOUT
import pytest
from dummyserver.server import HAS_IPV6
from dummyserver.testcase import HTTPDummyServerTestCase, IPv6HTTPDummyServerTestCase
from urllib3.connectionpool import port_by_scheme
from urllib3.exceptions import MaxRetryError, URLSchemeUnknown
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry
@pytest.mark.parametrize(['target', 'expected_target'], [('/echo_uri?q=1#fragment', b'/echo_uri?q=1'), ('/echo_uri?#', b'/echo_uri?'), ('/echo_uri#?', b'/echo_uri'), ('/echo_uri#?#', b'/echo_uri'), ('/echo_uri??#', b'/echo_uri??'), ('/echo_uri?%3f#', b'/echo_uri?%3F'), ('/echo_uri?%3F#', b'/echo_uri?%3F'), ('/echo_uri?[]', b'/echo_uri?%5B%5D')])
def test_encode_http_target(self, target, expected_target):
    with PoolManager() as http:
        url = 'http://%s:%d%s' % (self.host, self.port, target)
        r = http.request('GET', url)
        assert r.data == expected_target