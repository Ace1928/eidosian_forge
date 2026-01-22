import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_set_machine_power_reboot_failure(self):
    self.register_uris([dict(method='PUT', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid'], 'states', 'power']), status_code=400, json={'error': 'invalid'}, validate=dict(json={'target': 'rebooting'}))])
    self.assertRaises(exceptions.SDKException, self.cloud.set_machine_power_reboot, self.fake_baremetal_node['uuid'])
    self.assert_calls()