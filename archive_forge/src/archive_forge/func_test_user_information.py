import uuid
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystonemiddleware.auth_token import _base
from keystonemiddleware.tests.unit.auth_token import base
def test_user_information(self):
    token_id, token = self.get_token()
    plugin = self.get_plugin(token_id)
    self.assertTokenDataEqual(token_id, token, plugin.user)
    self.assertFalse(plugin.has_service_token)
    self.assertIsNone(plugin.service)