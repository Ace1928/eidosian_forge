import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
def test_zero_3(self):
    tv = 'A3 B9 B0 38 59 00 CE 76 1F 22 AE D5 48 E7 54 DA\n        10 A5 24 2D 62 E8 C6 58 E3 F3 A9 23 A7 55 56 07'
    btv = txt2bin(tv)
    res = TurboSHAKE128.new().read(10032)[-32:]
    self.assertEqual(res, btv)