import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
def test_ptn_17_5(self):
    tv = 'AD D5 3B 06 54 3E 58 4B 58 23 F6 26 99 6A EE 50\n        FE 45 ED 15 F2 02 43 A7 16 54 85 AC B4 AA 76 B4\n        FF DA 75 CE DF 6D 8C DC 95 C3 32 BD 56 F4 B9 86\n        B5 8B B1 7D 17 78 BF C1 B1 A9 75 45 CD F4 EC 9F'
    btv = txt2bin(tv)
    data = ptn(17 ** 5)
    res = TurboSHAKE256.new(data=data).read(64)
    self.assertEqual(res, btv)
    xof = TurboSHAKE256.new()
    for chunk in chunked(data, 8192):
        xof.update(chunk)
    res = xof.read(64)
    self.assertEqual(res, btv)