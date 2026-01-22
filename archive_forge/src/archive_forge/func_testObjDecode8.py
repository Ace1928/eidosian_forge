import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testObjDecode8(self):
    der = DerObject(2)
    self.assertEqual(der, der.decode(b('\x02\x02\x01\x02')))