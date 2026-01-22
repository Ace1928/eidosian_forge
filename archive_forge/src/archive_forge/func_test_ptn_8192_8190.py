import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import KangarooTwelve as K12
from Cryptodome.Util.py3compat import b, bchr
def test_ptn_8192_8190(self):
    tv = '6A 7C 1B 6A 5C D0 D8 C9 CA 94 3A 4A 21 6C C6 46\n        04 55 9A 2E A4 5F 78 57 0A 15 25 3D 67 BA 00 AE'
    btv = txt2bin(tv)
    res = K12.new(data=ptn(8192), custom=ptn(8190)).read(32)
    self.assertEqual(res, btv)