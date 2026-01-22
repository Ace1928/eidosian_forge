import tempfile
from tempest.lib import exceptions
from novaclient.tests.functional import base
from novaclient.tests.functional.v2 import fake_crypto
def test_list_keypair(self):
    key_name = self._create_keypair()
    keypairs = self._list_keypairs()
    self.assertIn(key_name, keypairs)