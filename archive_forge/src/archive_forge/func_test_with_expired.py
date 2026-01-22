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
def test_with_expired(self):
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    d = copy.deepcopy(self.TEST_RESPONSE_DICT)
    d['token']['expires_at'] = '2000-01-01T00:00:10.000123Z'
    a = v3.Password(self.TEST_URL, username='username', password='password')
    a.auth_ref = access.create(body=d)
    s = session.Session(auth=a)
    self.assertEqual({'X-Auth-Token': self.TEST_TOKEN}, s.get_auth_headers())
    self.assertEqual(a.auth_ref._data['token']['expires_at'], self.TEST_RESPONSE_DICT['token']['expires_at'])