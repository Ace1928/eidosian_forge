import requests
from unittest import mock
from keystoneclient import httpclient
from keystoneclient.tests.unit import utils
@mock.patch.object(requests, 'request')
def test_post_auth(self, MOCK_REQUEST):
    MOCK_REQUEST.return_value = FAKE_RESPONSE
    with self.deprecations.expect_deprecations_here():
        cl = httpclient.HTTPClient(username='username', password='password', project_id='tenant', auth_url='auth_test', cacert='ca.pem', cert=('cert.pem', 'key.pem'))
    cl.management_url = 'https://127.0.0.1:5000'
    cl.auth_token = 'token'
    with self.deprecations.expect_deprecations_here():
        cl.post('/hi', body=[1, 2, 3])
    mock_args, mock_kwargs = MOCK_REQUEST.call_args
    self.assertEqual(mock_args[0], 'POST')
    self.assertEqual(mock_args[1], REQUEST_URL)
    self.assertEqual(mock_kwargs['data'], '[1, 2, 3]')
    self.assertEqual(mock_kwargs['headers']['X-Auth-Token'], 'token')
    self.assertEqual(mock_kwargs['cert'], ('cert.pem', 'key.pem'))
    self.assertEqual(mock_kwargs['verify'], 'ca.pem')