from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_floating_ip_existing(self):
    self.register_uris([dict(method='DELETE', uri=self.get_mock_url('compute', append=['os-floating-ips', 'a-wild-id-appears'])), dict(method='GET', uri=self.get_mock_url('compute', append=['os-floating-ips']), json={'floating_ips': []})])
    ret = self.cloud.delete_floating_ip(floating_ip_id='a-wild-id-appears')
    self.assertTrue(ret)
    self.assert_calls()