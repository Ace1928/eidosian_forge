import json
import unittest
from binascii import unhexlify, hexlify
from .common import make_mac_tests
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import Poly1305
from Cryptodome.Cipher import AES, ChaCha20
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Util.strxor import strxor_c
def test_hexverify(self):
    h = Poly1305.new(key=self.key, cipher=AES)
    mac = h.hexdigest()
    h.hexverify(mac)
    self.assertRaises(ValueError, h.hexverify, '4556')