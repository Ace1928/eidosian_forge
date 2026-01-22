from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_floating_ip(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('compute', append=['os-floating-ips']), json={'floating_ips': self.mock_floating_ip_list_rep})])
    floating_ip = self.cloud.get_floating_ip(id='29')
    self.assertIsInstance(floating_ip, dict)
    self.assertEqual('198.51.100.29', floating_ip['floating_ip_address'])
    self.assert_calls()