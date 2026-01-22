import copy
import datetime
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from testtools import testcase
from keystoneclient import exceptions
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
def test_allow_override_of_auth_token(self):
    fake_url = '/fake-url'
    fake_token = 'fake_token'
    fake_resp = {'result': True}
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    self.stub_url('GET', [fake_url], json=fake_resp, base_url=self.TEST_ADMIN_IDENTITY_ENDPOINT)
    with self.deprecations.expect_deprecations_here():
        cl = client.Client(username='exampleuser', password='password', project_name='exampleproject', auth_url=self.TEST_URL)
    self.assertEqual(cl.auth_token, self.TEST_TOKEN)
    resp, body = cl._adapter.get(fake_url)
    self.assertEqual(fake_resp, body)
    token = self.requests_mock.last_request.headers.get('X-Auth-Token')
    self.assertEqual(self.TEST_TOKEN, token)
    cl.auth_token = fake_token
    resp, body = cl._adapter.get(fake_url)
    self.assertEqual(fake_resp, body)
    token = self.requests_mock.last_request.headers.get('X-Auth-Token')
    self.assertEqual(fake_token, token)
    del cl.auth_token
    resp, body = cl._adapter.get(fake_url)
    self.assertEqual(fake_resp, body)
    token = self.requests_mock.last_request.headers.get('X-Auth-Token')
    self.assertEqual(self.TEST_TOKEN, token)