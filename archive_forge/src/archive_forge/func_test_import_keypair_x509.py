from novaclient.tests.functional import base
from novaclient.tests.functional.v2 import fake_crypto
from novaclient.tests.functional.v2.legacy import test_keypairs
def test_import_keypair_x509(self):
    certif, fingerprint = fake_crypto.get_x509_cert_and_fingerprint()
    pub_key_file = self._create_public_key_file(certif)
    keypair = self._test_import_keypair(fingerprint, key_type='x509', pub_key=pub_key_file)
    self.assertIn('x509', keypair)