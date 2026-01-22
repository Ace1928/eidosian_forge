import httplib
import pytest
import StringIO
from mock import patch
from ..test_no_ssl import TestWithoutSSL
@pytest.mark.xfail(reason='This is not yet supported by urlfetch, presence of the ssl module will bypass urlfetch.')
def test_urlfetch_called_with_https(self):
    """
        Check that URLFetch is used when fetching https resources
        """
    resp = MockResponse('OK', 200, False, 'https://www.google.com', {'content-type': 'text/plain'})
    fetch_patch = patch('google.appengine.api.urlfetch.fetch', return_value=resp)
    with fetch_patch as fetch_mock:
        import urllib3
        pool = urllib3.HTTPSConnectionPool('www.google.com', '443')
        pool.ConnectionCls = urllib3.connection.UnverifiedHTTPSConnection
        r = pool.request('GET', '/')
        assert r.status == 200, r.data
        assert fetch_mock.call_count == 1