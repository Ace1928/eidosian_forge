import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
def test_new_domain(self):
    xof1 = self.TurboSHAKE.new(domain=29)
    xof2 = self.TurboSHAKE.new(domain=32)
    self.assertNotEqual(xof1.read(128), xof2.read(128))