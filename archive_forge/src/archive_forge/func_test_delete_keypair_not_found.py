import fixtures
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_keypair_not_found(self):
    self.register_uris([dict(method='DELETE', uri=self.get_mock_url('compute', 'public', append=['os-keypairs', self.keyname]), status_code=404)])
    self.assertFalse(self.cloud.delete_keypair(self.keyname))
    self.assert_calls()