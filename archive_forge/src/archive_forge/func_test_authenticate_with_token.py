import copy
import json
import time
import unittest
import uuid
from keystoneauth1 import _utils as ksa_utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.exceptions import ClientException
from keystoneauth1 import fixture
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import base as v3_base
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_authenticate_with_token(self):
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    a = v3.Token(self.TEST_URL, self.TEST_TOKEN)
    s = session.Session(auth=a)
    self.assertEqual({'X-Auth-Token': self.TEST_TOKEN}, s.get_auth_headers())
    req = {'auth': {'identity': {'methods': ['token'], 'token': {'id': self.TEST_TOKEN}}}}
    self.assertRequestBodyIs(json=req)
    self.assertRequestHeaderEqual('Content-Type', 'application/json')
    self.assertRequestHeaderEqual('Accept', 'application/json')
    self.assertEqual(s.auth.auth_ref.auth_token, self.TEST_TOKEN)