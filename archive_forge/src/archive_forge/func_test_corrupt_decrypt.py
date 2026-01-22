import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Cipher import ChaCha20_Poly1305
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
def test_corrupt_decrypt(self, tv):
    self._id = 'Wycheproof Corrupt Decrypt ChaCha20-Poly1305 Test #' + str(tv.id)
    if len(tv.iv) == 0 or len(tv.ct) < 1:
        return
    cipher = ChaCha20_Poly1305.new(key=tv.key, nonce=tv.iv)
    cipher.update(tv.aad)
    ct_corrupt = strxor(tv.ct, b'\x00' * (len(tv.ct) - 1) + b'\x01')
    self.assertRaises(ValueError, cipher.decrypt_and_verify, ct_corrupt, tv.tag)