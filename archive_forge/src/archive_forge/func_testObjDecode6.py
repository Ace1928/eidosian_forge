import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testObjDecode6(self):
    der = DerObject()
    der.decode(b('e\x01\x88'))
    self.assertEqual(der._tag_octet, 101)
    self.assertEqual(der.payload, b('\x88'))