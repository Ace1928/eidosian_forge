import copy
import uuid
from oslo_serialization import jsonutils
from keystoneauth1 import session as auth_session
from keystoneclient.auth import token_endpoint
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit.v3 import client_fixtures
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import client
def test_domain_scoped_init(self):
    token = client_fixtures.domain_scoped_token()
    self.stub_auth(json=token)
    with self.deprecations.expect_deprecations_here():
        c = client.Client(user_id=token.user_id, password='password', domain_name=token.domain_name, auth_url=self.TEST_URL)
    self.assertIsNotNone(c.auth_ref)
    self.assertTrue(c.auth_ref.domain_scoped)
    self.assertFalse(c.auth_ref.project_scoped)
    self.assertEqual(token.user_id, c.auth_user_id)
    self.assertEqual(token.domain_id, c.auth_domain_id)