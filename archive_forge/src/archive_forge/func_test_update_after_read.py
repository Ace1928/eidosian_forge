import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
def test_update_after_read(self):
    xof1 = self.TurboSHAKE.new()
    xof1.update(b'rrrr')
    xof1.read(90)
    self.assertRaises(TypeError, xof1.update, b'ttt')