import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_set_machine_maintenace_state(self):
    self.register_uris([dict(method='PUT', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid'], 'maintenance']), validate=dict(json={'reason': 'no reason'})), dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node)])
    self.cloud.set_machine_maintenance_state(self.fake_baremetal_node['uuid'], True, reason='no reason')
    self.assert_calls()