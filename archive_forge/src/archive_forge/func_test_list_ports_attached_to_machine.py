import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_ports_attached_to_machine(self):
    vif_id = '953ccbee-e854-450f-95fe-fe5e40d611ec'
    fake_port = {'id': vif_id, 'name': 'test'}
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node), dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid'], 'vifs']), json={'vifs': [{'id': vif_id}]}), dict(method='GET', uri=self.get_mock_url(service_type='network', resource='ports', base_url_append='v2.0', append=[vif_id]), json=fake_port)])
    res = self.cloud.list_ports_attached_to_machine(self.fake_baremetal_node['uuid'])
    self.assert_calls()
    self.assertEqual([_port.Port(**fake_port).to_dict(computed=False)], [i.to_dict(computed=False) for i in res])