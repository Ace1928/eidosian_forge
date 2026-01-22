from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_floating_ip_not_found(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('compute', append=['os-floating-ips']), json={'floating_ips': self.mock_floating_ip_list_rep})])
    floating_ip = self.cloud.get_floating_ip(id='666')
    self.assertIsNone(floating_ip)
    self.assert_calls()