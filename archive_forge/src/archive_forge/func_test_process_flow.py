import datetime
from unittest import mock
import uuid
from keystoneauth1 import fixture
import testtools
import webob
from keystonemiddleware import auth_token
from keystonemiddleware.auth_token import _request
@mock.patch.multiple(auth_token.BaseAuthProtocol, process_request=mock.DEFAULT, process_response=mock.DEFAULT)
def test_process_flow(self, process_request, process_response):
    m = auth_token.BaseAuthProtocol(FakeApp())
    process_request.return_value = None
    process_response.side_effect = lambda x: x
    req = webob.Request.blank('/', method='GET')
    resp = req.get_response(m)
    self.assertEqual(200, resp.status_code)
    self.assertEqual(1, process_request.call_count)
    self.assertIsInstance(process_request.call_args[0][0], _request._AuthTokenRequest)
    self.assertEqual(1, process_response.call_count)
    self.assertIsInstance(process_response.call_args[0][0], webob.Response)