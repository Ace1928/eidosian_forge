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
def test_redirect_302_no_location(self):
    fake = fakes.FakeHTTPResponse(302, 'OK', {}, '')
    self.request.side_effect = [(fake, '')]
    client = http.SessionClient(session=mock.ANY, auth=mock.ANY)
    e = self.assertRaises(exc.InvalidEndpoint, client.request, '', 'GET', redirect=True)
    self.assertEqual('Location not returned with 302', str(e))