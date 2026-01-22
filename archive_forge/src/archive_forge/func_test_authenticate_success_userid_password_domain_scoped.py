from oslo_serialization import jsonutils
from testtools import testcase
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import client
def test_authenticate_success_userid_password_domain_scoped(self):
    ident = self.TEST_REQUEST_BODY['auth']['identity']
    del ident['password']['user']['domain']
    del ident['password']['user']['name']
    ident['password']['user']['id'] = self.TEST_USER
    scope = self.TEST_REQUEST_BODY['auth']['scope']
    del scope['project']
    scope['domain'] = {}
    scope['domain']['id'] = self.TEST_DOMAIN_ID
    token = self.TEST_RESPONSE_DICT['token']
    del token['project']
    token['domain'] = {}
    token['domain']['id'] = self.TEST_DOMAIN_ID
    token['domain']['name'] = self.TEST_DOMAIN_NAME
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    with self.deprecations.expect_deprecations_here():
        cs = client.Client(user_id=self.TEST_USER, password=self.TEST_TOKEN, domain_id=self.TEST_DOMAIN_ID, auth_url=self.TEST_URL)
    self.assertEqual(cs.auth_domain_id, self.TEST_DOMAIN_ID)
    self.assertEqual(cs.management_url, self.TEST_RESPONSE_DICT['token']['catalog'][3]['endpoints'][2]['url'])
    self.assertEqual(cs.auth_token, self.TEST_RESPONSE_HEADERS['X-Subject-Token'])
    self.assertRequestBodyIs(json=self.TEST_REQUEST_BODY)