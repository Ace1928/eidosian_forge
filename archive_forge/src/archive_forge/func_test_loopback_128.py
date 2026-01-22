import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
def test_loopback_128(self):
    cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
    pt = get_tag_random('plaintext', 16 * 100)
    ct, mac = cipher.encrypt_and_digest(pt)
    cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
    pt2 = cipher.decrypt_and_verify(ct, mac)
    self.assertEqual(pt, pt2)