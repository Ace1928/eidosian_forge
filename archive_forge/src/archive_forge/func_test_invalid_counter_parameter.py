import unittest
from binascii import hexlify, unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128, SHA256
from Cryptodome.Util import Counter
def test_invalid_counter_parameter(self):
    self.assertRaises(TypeError, DES3.new, self.key_192, AES.MODE_CTR)
    self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_CTR, self.ctr_128)