import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testObjDecode5(self):
    der = DerObject(2)
    self.assertRaises(ValueError, der.decode, b('\x03\x02\x01\x02'))