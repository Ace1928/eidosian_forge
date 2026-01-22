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
def test_invalidate_response(self):
    auth_responses = [{'status_code': 200, 'json': self.TEST_RESPONSE_DICT, 'headers': {'X-Subject-Token': 'token1'}}, {'status_code': 200, 'json': self.TEST_RESPONSE_DICT, 'headers': {'X-Subject-Token': 'token2'}}]
    self.requests_mock.post('%s/auth/tokens' % self.TEST_URL, auth_responses)
    a = v3.Password(self.TEST_URL, username=self.TEST_USER, password=self.TEST_PASS)
    s = session.Session(auth=a)
    self.assertEqual('token1', s.get_token())
    self.assertEqual({'X-Auth-Token': 'token1'}, s.get_auth_headers())
    a.invalidate()
    self.assertEqual('token2', s.get_token())
    self.assertEqual({'X-Auth-Token': 'token2'}, s.get_auth_headers())