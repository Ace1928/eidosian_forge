import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testErrDecode3(self):
    der = DerSequence()
    self.assertRaises(ValueError, der.decode, b('0\x04\x02\x01\x01\x00'))
    self.assertRaises(ValueError, der.decode, b('0\x81\x03\x02\x01\x01'))
    self.assertRaises(ValueError, der.decode, b('0\x04\x02\x81\x01\x01'))