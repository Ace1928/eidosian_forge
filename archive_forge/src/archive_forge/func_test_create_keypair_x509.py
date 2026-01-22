from novaclient.tests.functional import base
from novaclient.tests.functional.v2 import fake_crypto
from novaclient.tests.functional.v2.legacy import test_keypairs
def test_create_keypair_x509(self):
    key_name = self._create_keypair(key_type='x509')
    keypair = self._show_keypair(key_name)
    self.assertIn(key_name, keypair)
    self.assertIn('x509', keypair)