from openstack.tests import fakes
from openstack.tests.unit import base
def test_detach_ip_from_server(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('compute', append=['os-floating-ips']), json={'floating_ips': self.mock_floating_ip_list_rep}), dict(method='POST', uri=self.get_mock_url('compute', append=['servers', self.fake_server['id'], 'action']), validate=dict(json={'removeFloatingIp': {'address': '203.0.113.1'}}))])
    self.cloud.detach_ip_from_server(server_id='server-id', floating_ip_id=1)
    self.assert_calls()