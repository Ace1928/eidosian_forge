import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testEncode4(self):
    der = DerBoolean(False, explicit=5)
    self.assertEqual(der.encode(), b'\xa5\x03\x01\x01\x00')