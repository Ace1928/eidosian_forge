from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from oslo_context import context
from castellan.common import exception
from castellan.common.objects import symmetric_key as sym_key
from castellan.tests.unit.key_manager import mock_key_manager as mock_key_mgr
from castellan.tests.unit.key_manager import test_key_manager as test_key_mgr
def test_store_key_and_get_metadata(self):
    secret_key = bytes(b'0' * 64)
    _key = sym_key.SymmetricKey('AES', 64 * 8, secret_key)
    key_id = self.key_mgr.store(self.context, _key)
    actual_key = self.key_mgr.get(self.context, key_id, metadata_only=True)
    self.assertIsNone(actual_key.get_encoded())
    self.assertTrue(actual_key.is_metadata_only())
    self.assertIsNotNone(actual_key.id)