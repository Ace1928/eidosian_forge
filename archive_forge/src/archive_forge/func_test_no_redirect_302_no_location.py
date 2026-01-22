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
def test_no_redirect_302_no_location(self):
    fake = fakes.FakeHTTPResponse(302, 'OK', {'location': 'http://no.where/ishere'}, '')
    self.request.side_effect = [(fake, '')]
    client = http.SessionClient(session=mock.ANY, auth=mock.ANY)
    self.assertEqual(fake, client.request('', 'GET'))