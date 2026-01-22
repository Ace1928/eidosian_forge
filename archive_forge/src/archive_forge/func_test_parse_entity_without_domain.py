from unittest import mock
from keystoneauth1 import exceptions as keystone_exceptions
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import keystone_constraints as ks_constr
from heat.tests import common
@mock.patch.object(keystone.KeystoneClientPlugin, 'client')
def test_parse_entity_without_domain(self, client_keystone):
    client_keystone.return_value = self._client
    client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
    client_plugin.get_domain_id = mock.MagicMock()
    client_plugin.get_domain_id.return_value = self.sample_uuid
    self.assertEqual(client_plugin.parse_entity_with_domain('entity', 'entity_type'), ('entity', None))