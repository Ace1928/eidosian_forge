import uuid
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystonemiddleware.auth_token import _base
from keystonemiddleware.tests.unit.auth_token import base
def test_with_service_information(self):
    token_id, token = self.get_token()
    service_id, service = self.get_token(service=True)
    plugin = self.get_plugin(token_id, service_id)
    self.assertTokenDataEqual(token_id, token, plugin.user)
    self.assertTokenDataEqual(service_id, service, plugin.service)