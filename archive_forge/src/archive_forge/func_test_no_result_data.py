from unittest import mock
from oslo_serialization import jsonutils
import requests
import webob
from keystonemiddleware import ec2_token
from keystonemiddleware.tests.unit import utils
@mock.patch.object(requests, 'post', return_value=FakeResponse(EMPTY_RESPONSE))
def test_no_result_data(self, mock_request):
    req = webob.Request.blank('/test')
    req.GET['Signature'] = 'test-signature'
    req.GET['AWSAccessKeyId'] = 'test-key-id'
    resp = req.get_response(self.middleware)
    self._validate_ec2_error(resp, 400, 'AuthFailure')
    mock_request.assert_called_with('http://localhost:5000/v3/ec2tokens', data=mock.ANY, headers=mock.ANY, verify=mock.ANY, cert=mock.ANY, timeout=mock.ANY)