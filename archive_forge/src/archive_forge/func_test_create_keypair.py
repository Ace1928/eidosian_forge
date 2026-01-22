import fixtures
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_keypair(self):
    self.register_uris([dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['os-keypairs']), json={'keypair': self.key}, validate=dict(json={'keypair': {'name': self.key['name'], 'public_key': self.key['public_key']}}))])
    new_key = self.cloud.create_keypair(self.keyname, self.key['public_key'])
    new_key_cmp = new_key.to_dict(ignore_none=True)
    new_key_cmp.pop('location')
    new_key_cmp.pop('id')
    self.assertEqual(new_key_cmp, self.key)
    self.assert_calls()