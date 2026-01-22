import copy
import json
import time
import uuid
from keystoneauth1 import _utils as ksa_utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.identity import v2
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_authenticate_with_user_id_password(self):
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    a = v2.Password(self.TEST_URL, user_id=self.TEST_USER, password=self.TEST_PASS)
    self.assertIsNone(a.username)
    self.assertFalse(a.has_scope_parameters)
    s = session.Session(a)
    self.assertEqual({'X-Auth-Token': self.TEST_TOKEN}, s.get_auth_headers())
    req = {'auth': {'passwordCredentials': {'userId': self.TEST_USER, 'password': self.TEST_PASS}}}
    self.assertRequestBodyIs(json=req)
    self.assertRequestHeaderEqual('Content-Type', 'application/json')
    self.assertRequestHeaderEqual('Accept', 'application/json')
    self.assertEqual(s.auth.auth_ref.auth_token, self.TEST_TOKEN)