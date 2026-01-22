import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import KangarooTwelve as K12
from Cryptodome.Util.py3compat import b, bchr
def test_ptn_c_41(self):
    tv = 'D8 48 C5 06 8C ED 73 6F 44 62 15 9B 98 67 FD 4C\n        20 B8 08 AC C3 D5 BC 48 E0 B0 6B A0 A3 76 2E C4'
    btv = txt2bin(tv)
    custom = ptn(41)
    res = K12.new(data=b'\xff', custom=custom).read(32)
    self.assertEqual(res, btv)