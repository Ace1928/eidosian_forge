import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
def test_ptn_17_2(self):
    tv = '66 B8 10 DB 8E 90 78 04 24 C0 84 73 72 FD C9 57\n        10 88 2F DE 31 C6 DF 75 BE B9 D4 CD 93 05 CF CA\n        E3 5E 7B 83 E8 B7 E6 EB 4B 78 60 58 80 11 63 16\n        FE 2C 07 8A 09 B9 4A D7 B8 21 3C 0A 73 8B 65 C0'
    btv = txt2bin(tv)
    res = TurboSHAKE256.new(data=ptn(17 ** 2)).read(64)
    self.assertEqual(res, btv)