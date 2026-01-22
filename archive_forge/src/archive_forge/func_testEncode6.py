import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testEncode6(self):
    der = DerSequence()
    der.append(384)
    der.append(255)
    self.assertEqual(der.encode(), b('0\x08\x02\x02\x01\x80\x02\x02\x00ÿ'))
    self.assertTrue(der.hasOnlyInts())
    self.assertTrue(der.hasOnlyInts(False))
    der = DerSequence()
    der.append(2)
    der.append(-2)
    self.assertEqual(der.encode(), b('0\x06\x02\x01\x02\x02\x01þ'))
    self.assertEqual(der.hasInts(), 1)
    self.assertEqual(der.hasInts(False), 2)
    self.assertFalse(der.hasOnlyInts())
    self.assertTrue(der.hasOnlyInts(False))
    der.append(1)
    der[1:] = [9, 8]
    self.assertEqual(len(der), 3)
    self.assertEqual(der[1:], [9, 8])
    self.assertEqual(der[1:-1], [9])
    self.assertEqual(der.encode(), b('0\t\x02\x01\x02\x02\x01\t\x02\x01\x08'))