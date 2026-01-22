import copy
from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.keystone import service
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_service_handle_create(self):
    rsrc = self._setup_service_resource('test_service_create')
    mock_service = self._get_mock_service()
    self.services.create.return_value = mock_service
    self.assertEqual('test_service_1', rsrc.properties.get(service.KeystoneService.NAME))
    self.assertEqual('Test service', rsrc.properties.get(service.KeystoneService.DESCRIPTION))
    self.assertEqual('orchestration', rsrc.properties.get(service.KeystoneService.TYPE))
    self.assertFalse(rsrc.properties.get(service.KeystoneService.ENABLED))
    rsrc.handle_create()
    self.services.create.assert_called_once_with(name='test_service_1', description='Test service', type='orchestration', enabled=False)
    self.assertEqual(mock_service.id, rsrc.resource_id)