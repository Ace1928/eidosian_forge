import fixtures
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_keypair_exception(self):
    self.register_uris([dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['os-keypairs']), status_code=400, validate=dict(json={'keypair': {'name': self.key['name'], 'public_key': self.key['public_key']}}))])
    self.assertRaises(exceptions.SDKException, self.cloud.create_keypair, self.keyname, self.key['public_key'])
    self.assert_calls()