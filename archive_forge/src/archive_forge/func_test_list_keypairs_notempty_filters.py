import fixtures
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_keypairs_notempty_filters(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['os-keypairs'], qs_elements=['user_id=b']), json={'keypairs': [{'keypair': self.key}]})])
    keypairs = self.cloud.list_keypairs(filters={'user_id': 'b', 'fake': 'dummy'})
    self.assertEqual(len(keypairs), 1)
    self.assertEqual(keypairs[0].name, self.key['name'])
    self.assert_calls()