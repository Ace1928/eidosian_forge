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
def test_doesnt_log_password(self):
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    password = uuid.uuid4().hex
    a = v3.Password(self.TEST_URL, username=self.TEST_USER, password=password)
    s = session.Session(a)
    self.assertEqual(self.TEST_TOKEN, s.get_token())
    self.assertEqual({'X-Auth-Token': self.TEST_TOKEN}, s.get_auth_headers())
    self.assertNotIn(password, self.logger.output)