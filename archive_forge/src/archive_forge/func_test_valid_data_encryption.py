import fixtures
import hashlib
import uuid
from oslo_log import log
from keystone.common import fernet_utils
from keystone.credential.providers import fernet as credential_fernet
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
def test_valid_data_encryption(self):
    blob = uuid.uuid4().hex
    encrypted_blob, primary_key_hash = self.provider.encrypt(blob)
    decrypted_blob = self.provider.decrypt(encrypted_blob)
    self.assertNotEqual(blob, encrypted_blob)
    self.assertEqual(blob, decrypted_blob)
    self.assertIsNotNone(primary_key_hash)