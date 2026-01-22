import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testDecode4(self):
    der = DerBoolean(explicit=5)
    der.decode(b'\xa5\x03\x01\x01\x00')
    self.assertEqual(der.value, False)