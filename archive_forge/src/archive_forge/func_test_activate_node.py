import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_activate_node(self):
    self.fake_baremetal_node['provision_state'] = 'active'
    self.register_uris([dict(method='PUT', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid'], 'states', 'provision']), validate=dict(json={'target': 'active', 'configdrive': 'http://host/file'})), dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node)])
    return_value = self.cloud.activate_node(self.fake_baremetal_node['uuid'], configdrive='http://host/file', wait=True)
    self.assertIsNone(return_value)
    self.assert_calls()