import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testEncode5(self):
    der = DerSequence()
    der += 1
    der += b('0\x00')
    self.assertEqual(der.encode(), b('0\x05\x02\x01\x010\x00'))