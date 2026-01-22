from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests.unit import base
def test_search_ports_not_found(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports']), json=self.mock_neutron_port_list_rep)])
    ports = self.cloud.search_ports(name_or_id='non-existent')
    self.assertEqual(0, len(ports))
    self.assert_calls()