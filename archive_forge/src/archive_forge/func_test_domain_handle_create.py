from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import resource
from heat.engine.resources.openstack.keystone import domain
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_domain_handle_create(self):
    mock_domain = self._get_mock_domain()
    self.domains.create.return_value = mock_domain
    self.assertEqual('test_domain_1', self.test_domain.properties.get(domain.KeystoneDomain.NAME))
    self.assertEqual('Test domain', self.test_domain.properties.get(domain.KeystoneDomain.DESCRIPTION))
    self.assertEqual(True, self.test_domain.properties.get(domain.KeystoneDomain.ENABLED))
    self.test_domain.handle_create()
    self.domains.create.assert_called_once_with(name='test_domain_1', description='Test domain', enabled=True)
    self.assertEqual(mock_domain.id, self.test_domain.resource_id)