import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testErrorDecode1(self):
    der = DerBoolean()
    self.assertRaises(ValueError, der.decode, b'\x02\x01\x00')