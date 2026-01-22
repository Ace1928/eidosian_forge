import uuid
from oslo_serialization import jsonutils
from keystoneauth1 import fixture
from keystoneauth1 import session as auth_session
from keystoneclient.auth import token_endpoint
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit.v2_0 import client_fixtures
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
def test_auth_ref_load(self):
    self.stub_auth(json=client_fixtures.project_scoped_token())
    with self.deprecations.expect_deprecations_here():
        cl = client.Client(username='exampleuser', password='password', project_name='exampleproject', auth_url=self.TEST_URL)
    cache = jsonutils.dumps(cl.auth_ref)
    with self.deprecations.expect_deprecations_here():
        new_client = client.Client(auth_ref=jsonutils.loads(cache))
    self.assertIsNotNone(new_client.auth_ref)
    with self.deprecations.expect_deprecations_here():
        self.assertTrue(new_client.auth_ref.scoped)
    self.assertTrue(new_client.auth_ref.project_scoped)
    self.assertFalse(new_client.auth_ref.domain_scoped)
    self.assertIsNone(new_client.auth_ref.trust_id)
    self.assertFalse(new_client.auth_ref.trust_scoped)
    self.assertEqual(new_client.username, 'exampleuser')
    self.assertIsNone(new_client.password)
    self.assertEqual(new_client.management_url, 'http://admin:35357/v2.0')