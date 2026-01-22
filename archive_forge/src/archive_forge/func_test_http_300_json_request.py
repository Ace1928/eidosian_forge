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
def test_http_300_json_request(self, mock_request):
    mock_request.return_value = fakes.FakeHTTPResponse(300, 'OK', {'content-type': 'application/json'}, '{}')
    client = http.HTTPClient('http://example.com:8004')
    e = self.assertRaises(exc.HTTPMultipleChoices, client.json_request, 'GET', '')
    self.assertIsNotNone(str(e))
    mock_request.assert_called_with('GET', 'http://example.com:8004', allow_redirects=False, headers={'Content-Type': 'application/json', 'Accept': 'application/json', 'User-Agent': 'python-heatclient'})