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
def test_404_error_response(self):
    fake = fakes.FakeHTTPResponse(404, 'FAIL', {'content-type': 'application/octet-stream'}, '')
    self.request.return_value = (fake, '')
    client = http.SessionClient(session=mock.ANY, auth=mock.ANY)
    e = self.assertRaises(exc.HTTPNotFound, client.request, '', 'GET')
    self.assertIsNotNone(str(e))