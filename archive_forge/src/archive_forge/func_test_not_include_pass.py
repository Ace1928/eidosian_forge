import socket
from unittest import mock
import io
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
import testtools
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
from heatclient.tests.unit import fakes
def test_not_include_pass(self, mock_request):
    fake500 = fakes.FakeHTTPResponse(500, 'ERROR', {'content-type': 'application/octet-stream'}, b'(HTTP 401)')
    mock_request.return_value = fake500
    client = http.HTTPClient('http://example.com:8004')
    e = self.assertRaises(exc.HTTPUnauthorized, client.raw_request, 'GET', '')
    self.assertIn('Authentication failed', str(e))
    mock_request.assert_called_with('GET', 'http://example.com:8004', allow_redirects=False, headers={'Content-Type': 'application/octet-stream', 'User-Agent': 'python-heatclient'})