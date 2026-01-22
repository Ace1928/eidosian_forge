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
def test_302_location_no_endpoint(self):
    fake1 = fakes.FakeHTTPResponse(302, 'OK', {'location': 'http://no.where/ishere'}, '')
    fake2 = fakes.FakeHTTPResponse(200, 'OK', {'content-type': 'application/json'}, jsonutils.dumps({'Mount': 'Fuji'}))
    self.request.side_effect = [(fake1, ''), (fake2, jsonutils.dumps({'Mount': 'Fuji'}))]
    client = http.SessionClient(session=mock.ANY, auth=mock.ANY)
    resp = client.request('', 'GET', redirect=True)
    self.assertEqual(200, resp.status_code)
    self.assertEqual({'Mount': 'Fuji'}, utils.get_response_body(resp))
    self.assertEqual(('', 'GET'), self.request.call_args_list[0][0])
    self.assertEqual(('http://no.where/ishere', 'GET'), self.request.call_args_list[1][0])
    for call in self.request.call_args_list:
        self.assertEqual({'headers': {'Content-Type': 'application/json'}, 'user_agent': 'python-heatclient', 'raise_exc': False, 'redirect': True}, call[1])