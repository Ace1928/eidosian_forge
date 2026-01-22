from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from oslo_context import context
from castellan.common import exception
from castellan.common.objects import symmetric_key as sym_key
from castellan.tests.unit.key_manager import mock_key_manager as mock_key_mgr
from castellan.tests.unit.key_manager import test_key_manager as test_key_mgr
def test_create_key_pair_encryption(self):
    private_key_uuid, public_key_uuid = self.key_mgr.create_key_pair(self.context, 'RSA', 2048)
    private_key = self.key_mgr.get(self.context, private_key_uuid)
    public_key = self.key_mgr.get(self.context, public_key_uuid)
    crypto_private_key = get_cryptography_private_key(private_key)
    crypto_public_key = get_cryptography_public_key(public_key)
    message = b'secret plaintext'
    ciphertext = crypto_public_key.encrypt(message, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA1()), algorithm=hashes.SHA1(), label=None))
    plaintext = crypto_private_key.decrypt(ciphertext, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA1()), algorithm=hashes.SHA1(), label=None))
    self.assertEqual(message, plaintext)