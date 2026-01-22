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
def test_receipt_response_is_handled(self):
    self.stub_receipt()
    a = v3.Password(self.TEST_URL, username=self.TEST_USER, password=self.TEST_PASS, user_domain_id=self.TEST_DOMAIN_ID, project_id=self.TEST_TENANT_ID)
    s = session.Session(a)
    self.assertRaises(exceptions.MissingAuthMethods, s.get_auth_headers, None)