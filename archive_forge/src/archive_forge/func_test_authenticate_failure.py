import copy
import datetime
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from testtools import testcase
from keystoneclient import exceptions
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
def test_authenticate_failure(self):
    _auth = 'auth'
    _cred = 'passwordCredentials'
    _pass = 'password'
    self.TEST_REQUEST_BODY[_auth][_cred][_pass] = 'bad_key'
    error = {'unauthorized': {'message': 'Unauthorized', 'code': '401'}}
    self.stub_auth(status_code=401, json=error)
    with testcase.ExpectedException(exceptions.Unauthorized):
        with self.deprecations.expect_deprecations_here():
            client.Client(username=self.TEST_USER, password='bad_key', project_id=self.TEST_TENANT_ID, auth_url=self.TEST_URL)
    self.assertRequestBodyIs(json=self.TEST_REQUEST_BODY)