from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import resource
from heat.engine.resources.openstack.keystone import user
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_user_handle_create_default(self):
    values = {user.KeystoneUser.NAME: None, user.KeystoneUser.DESCRIPTION: self._get_property_schema_value_default(user.KeystoneUser.DESCRIPTION), user.KeystoneUser.DOMAIN: self._get_property_schema_value_default(user.KeystoneUser.DOMAIN), user.KeystoneUser.ENABLED: self._get_property_schema_value_default(user.KeystoneUser.ENABLED), user.KeystoneUser.ROLES: None, user.KeystoneUser.GROUPS: None, user.KeystoneUser.PASSWORD: 'password', user.KeystoneUser.EMAIL: 'abc@xyz.com', user.KeystoneUser.DEFAULT_PROJECT: 'default_project'}

    def _side_effect(key):
        return values[key]
    mock_user = self._get_mock_user()
    self.users.create.return_value = mock_user
    self.test_user.properties = mock.MagicMock()
    self.test_user.properties.get.side_effect = _side_effect
    self.test_user.properties.__getitem__.side_effect = _side_effect
    self.test_user.physical_resource_name = mock.MagicMock()
    self.test_user.physical_resource_name.return_value = 'foo'
    self.test_user.handle_create()
    self.users.create.assert_called_once_with(name='foo', description='', domain='default', enabled=True, email='abc@xyz.com', password='password', default_project='default_project')