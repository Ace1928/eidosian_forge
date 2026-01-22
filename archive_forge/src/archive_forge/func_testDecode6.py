import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testDecode6(self):
    der = DerSequence()
    der.decode(b('0\x08\x02\x02\x01\x80\x02\x02\x00ÿ'))
    self.assertEqual(len(der), 2)
    self.assertEqual(der[0], 384)
    self.assertEqual(der[1], 255)