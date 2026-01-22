import fixtures
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_keypairs_exception(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['os-keypairs']), status_code=400)])
    self.assertRaises(exceptions.SDKException, self.cloud.list_keypairs)
    self.assert_calls()