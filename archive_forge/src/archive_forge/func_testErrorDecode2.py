import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testErrorDecode2(self):
    der = DerBoolean()
    self.assertRaises(ValueError, der.decode, b'\x01\x01\x00\xff')