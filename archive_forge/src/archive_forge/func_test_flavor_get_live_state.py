from unittest import mock
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_flavor_get_live_state(self):
    self.create_flavor()
    value = mock.MagicMock()
    value.get_keys.return_value = {'key': 'value'}
    value.to_dict.return_value = {'ram': 1024, 'disk': 0, 'vcpus': 1, 'rxtx_factor': 1.0, 'OS-FLV-EXT-DATA:ephemeral': 0, 'os-flavor-access:is_public': True}
    self.flavors.get.return_value = value
    self.my_flavor.resource_id = '1234'
    reality = self.my_flavor.get_live_state(self.my_flavor.properties)
    self.assertEqual({'extra_specs': {'key': 'value'}}, reality)