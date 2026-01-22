from openstack.tests import fakes
from openstack.tests.unit import base
def test_add_ip_from_pool(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('compute', append=['os-floating-ips']), json={'floating_ips': self.mock_floating_ip_list_rep}), dict(method='GET', uri=self.get_mock_url('compute', append=['os-floating-ips']), json={'floating_ips': self.mock_floating_ip_list_rep}), dict(method='POST', uri=self.get_mock_url('compute', append=['servers', self.fake_server['id'], 'action']), validate=dict(json={'addFloatingIp': {'address': '203.0.113.1', 'fixed_address': '192.0.2.129'}}))])
    server = self.cloud._add_ip_from_pool(server=self.fake_server, network='nova', fixed_address='192.0.2.129')
    self.assertEqual(server, self.fake_server)
    self.assert_calls()