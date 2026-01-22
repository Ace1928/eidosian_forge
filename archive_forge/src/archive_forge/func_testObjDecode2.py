import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testObjDecode2(self):
    der = DerObject(2)
    der.decode(b('\x02\x81\x80' + '1' * 128))
    self.assertEqual(der.payload, b('1') * 128)
    self.assertEqual(der._tag_octet, 2)