import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testEncode3(self):
    der = DerBoolean(False, implicit=18)
    self.assertEqual(der.encode(), b'\x92\x01\x00')