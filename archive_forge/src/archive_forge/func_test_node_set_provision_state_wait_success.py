import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_node_set_provision_state_wait_success(self):
    self.fake_baremetal_node['provision_state'] = 'active'
    self.register_uris([dict(method='PUT', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid'], 'states', 'provision']), validate=dict(json={'target': 'active'})), dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node)])
    return_value = self.cloud.node_set_provision_state(self.fake_baremetal_node['uuid'], 'active', wait=True)
    self.assertSubdict(self.fake_baremetal_node, return_value)
    self.assert_calls()