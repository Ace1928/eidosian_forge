import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_attach_port_to_machine(self):
    vif_id = '953ccbee-e854-450f-95fe-fe5e40d611ec'
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node), dict(method='GET', uri=self.get_mock_url(service_type='network', resource='ports', base_url_append='v2.0', append=[vif_id]), json={'id': vif_id}), dict(method='POST', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid'], 'vifs']))])
    self.cloud.attach_port_to_machine(self.fake_baremetal_node['uuid'], vif_id)
    self.assert_calls()