import copy
from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.keystone import endpoint
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_endpoint_handle_create_default(self):
    rsrc = self._setup_endpoint_resource('test_create_with_defaults', use_default=True)
    mock_endpoint = self._get_mock_endpoint()
    self.endpoints.create.return_value = mock_endpoint
    rsrc.physical_resource_name = mock.MagicMock()
    rsrc.physical_resource_name.return_value = 'stack_endpoint_foo'
    self.assertIsNone(rsrc.properties.get(endpoint.KeystoneEndpoint.NAME))
    self.assertTrue(rsrc.properties.get(endpoint.KeystoneEndpoint.ENABLED))
    rsrc.handle_create()
    self.endpoints.create.assert_called_once_with(service='heat', url='http://127.0.0.1:8004/v1/tenant-id', interface='public', region='RegionOne', name='stack_endpoint_foo', enabled=True)