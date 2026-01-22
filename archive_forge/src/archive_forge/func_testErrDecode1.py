import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testErrDecode1(self):
    der = DerSetOf()
    self.assertRaises(ValueError, der.decode, b('1\x08\x02\x02\x01\x80\x02\x02\x00ÿª'))