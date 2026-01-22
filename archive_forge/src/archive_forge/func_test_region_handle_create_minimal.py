from unittest import mock
from urllib import parse
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import resource
from heat.engine.resources.openstack.keystone import region
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_region_handle_create_minimal(self):
    values = {'description': 'sample region', 'enabled': True, 'parent_region': None, 'id': None}

    def _side_effect(key):
        return values[key]
    mock_region = self._get_mock_region()
    self.regions.create.return_value = mock_region
    self.test_region.properties = mock.MagicMock()
    self.test_region.properties.__getitem__.side_effect = _side_effect
    self.test_region.handle_create()
    self.regions.create.assert_called_once_with(id=None, description='sample region', parent_region=None, enabled=True)