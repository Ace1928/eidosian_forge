import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testStrict1(self):
    number = DerInteger()
    number.decode(b'\x02\x02\x00\x01')
    number.decode(b'\x02\x02\x00\x7f')
    self.assertRaises(ValueError, number.decode, b'\x02\x02\x00\x01', strict=True)
    self.assertRaises(ValueError, number.decode, b'\x02\x02\x00\x7f', strict=True)