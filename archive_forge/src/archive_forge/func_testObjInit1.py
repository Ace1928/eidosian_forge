import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testObjInit1(self):
    self.assertRaises(ValueError, DerObject, b('\x00\x99'))
    self.assertRaises(ValueError, DerObject, 31)