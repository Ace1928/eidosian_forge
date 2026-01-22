import datetime
import uuid
from keystoneauth1 import fixture
from oslo_utils import timeutils
import testresources
from keystoneclient import access
from keystoneclient.tests.unit import client_fixtures as token_data
from keystoneclient.tests.unit.v2_0 import client_fixtures
from keystoneclient.tests.unit.v2_0 import utils
def test_override_auth_token(self):
    token = fixture.V2Token()
    token.set_scope()
    token.add_role()
    new_auth_token = uuid.uuid4().hex
    auth_ref = access.AccessInfo.factory(body=token)
    self.assertEqual(token.token_id, auth_ref.auth_token)
    auth_ref.auth_token = new_auth_token
    self.assertEqual(new_auth_token, auth_ref.auth_token)
    del auth_ref.auth_token
    self.assertEqual(token.token_id, auth_ref.auth_token)