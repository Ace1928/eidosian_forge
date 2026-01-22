from openstack.tests import fakes
from openstack.tests.functional import base
def test_create_and_delete_with_key(self):
    """Test creating and deleting keypairs functionality"""
    name = self.getUniqueString('keypair')
    self.addCleanup(self.user_cloud.delete_keypair, name)
    keypair = self.user_cloud.create_keypair(name=name, public_key=fakes.FAKE_PUBLIC_KEY)
    self.assertEqual(keypair['name'], name)
    self.assertIsNotNone(keypair['public_key'])
    self.assertIsNone(keypair['private_key'])
    self.assertIsNotNone(keypair['fingerprint'])
    self.assertEqual(keypair['type'], 'ssh')
    keypairs = self.user_cloud.list_keypairs()
    self.assertIn(name, [k['name'] for k in keypairs])
    self.user_cloud.delete_keypair(name)
    keypairs = self.user_cloud.list_keypairs()
    self.assertNotIn(name, [k['name'] for k in keypairs])