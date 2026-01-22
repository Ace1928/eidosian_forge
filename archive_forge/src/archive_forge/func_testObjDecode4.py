import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testObjDecode4(self):
    der = DerObject(2, constructed=False, implicit=15)
    self.assertRaises(ValueError, der.decode, b('\x02\x02\x01\x02'))
    der.decode(b('\x8f\x01\x00'))
    self.assertEqual(der.payload, b('\x00'))
    der = DerObject(2, constructed=True, implicit=15)
    self.assertRaises(ValueError, der.decode, b('\x02\x02\x01\x02'))
    der.decode(b('¯\x01\x00'))
    self.assertEqual(der.payload, b('\x00'))