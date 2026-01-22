import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, is_string
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
from Cryptodome.SelfTest.Cipher.test_CBC import BlockChainingTests
def test_segment_size_64(self):
    for bits in range(8, 65, 8):
        cipher = DES3.new(self.key_192, DES3.MODE_CFB, self.iv_64, segment_size=bits)
    for bits in (0, 7, 9, 63, 65):
        self.assertRaises(ValueError, DES3.new, self.key_192, AES.MODE_CFB, self.iv_64, segment_size=bits)