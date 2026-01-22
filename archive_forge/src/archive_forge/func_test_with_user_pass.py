import io
import uuid
from keystoneauth1 import fixture
from keystoneauth1 import plugin as ksa_plugin
from keystoneauth1 import session
from oslo_log import log as logging
from requests_mock.contrib import fixture as rm_fixture
from keystonemiddleware.auth_token import _auth
from keystonemiddleware.tests.unit import utils
def test_with_user_pass(self):
    base_uri = 'http://testhost:8888/admin'
    token = fixture.V2Token()
    admin_tenant_name = uuid.uuid4().hex
    self.requests_mock.post(base_uri + '/v2.0/tokens', json=token)
    plugin = self.new_plugin(identity_uri=base_uri, admin_user=uuid.uuid4().hex, admin_password=uuid.uuid4().hex, admin_tenant_name=admin_tenant_name)
    self.assertEqual(token.token_id, plugin.get_token(self.session))