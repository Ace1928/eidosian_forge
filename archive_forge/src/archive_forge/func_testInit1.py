import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testInit1(self):
    der = DerSetOf([DerInteger(1), DerInteger(2)])
    self.assertEqual(der.encode(), b('1\x06\x02\x01\x01\x02\x01\x02'))