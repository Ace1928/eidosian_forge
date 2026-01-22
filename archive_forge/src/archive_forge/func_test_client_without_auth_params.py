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
def test_client_without_auth_params(self):
    with self.deprecations.expect_deprecations_here():
        self.assertRaises(exceptions.AuthorizationFailure, client.Client, project_name='exampleproject', auth_url=self.TEST_URL)