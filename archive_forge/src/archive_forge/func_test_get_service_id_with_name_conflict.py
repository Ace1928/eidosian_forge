from unittest import mock
from keystoneauth1 import exceptions as keystone_exceptions
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import keystone_constraints as ks_constr
from heat.tests import common
@mock.patch.object(keystone.KeystoneClientPlugin, 'client')
def test_get_service_id_with_name_conflict(self, client_keystone):
    self._client.client.services.get.side_effect = keystone_exceptions.NotFound
    self._client.client.services.list.return_value = [self._get_mock_service(), self._get_mock_service()]
    client_keystone.return_value = self._client
    client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
    ex = self.assertRaises(exception.KeystoneServiceNameConflict, client_plugin.get_service_id, self.sample_name)
    msg = 'Keystone has more than one service with same name %s. Please use service id instead of name' % self.sample_name
    self.assertEqual(msg, str(ex))
    self.assertRaises(keystone_exceptions.NotFound, self._client.client.services.get, self.sample_name)
    self._client.client.services.list.assert_called_once_with(name=self.sample_name)