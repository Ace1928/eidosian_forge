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
def test_http_json_request_argument_passed_to_requests(self, mock_request):
    """Check that we have sent the proper arguments to requests."""
    mock_request.return_value = fakes.FakeHTTPResponse(200, 'OK', {'content-type': 'application/json'}, '{}')
    client = http.HTTPClient('http://example.com:8004')
    client.verify_cert = True
    client.cert_file = 'RANDOM_CERT_FILE'
    client.key_file = 'RANDOM_KEY_FILE'
    client.auth_url = 'http://AUTH_URL'
    resp, body = client.json_request('GET', '', data='text')
    self.assertEqual(200, resp.status_code)
    self.assertEqual({}, body)
    mock_request.assert_called_with('GET', 'http://example.com:8004', allow_redirects=False, cert=('RANDOM_CERT_FILE', 'RANDOM_KEY_FILE'), verify=True, data='"text"', headers={'Content-Type': 'application/json', 'Accept': 'application/json', 'X-Auth-Url': 'http://AUTH_URL', 'User-Agent': 'python-heatclient'})