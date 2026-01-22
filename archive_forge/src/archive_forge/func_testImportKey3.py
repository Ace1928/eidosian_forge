import unittest
import re
from Cryptodome.PublicKey import DSA
from Cryptodome.SelfTest.st_common import *
from Cryptodome.Util.py3compat import *
from binascii import unhexlify
def testImportKey3(self):
    key_obj = DSA.importKey(self.der_private)
    self.assertTrue(key_obj.has_private())
    self.assertEqual(self.y, key_obj.y)
    self.assertEqual(self.p, key_obj.p)
    self.assertEqual(self.q, key_obj.q)
    self.assertEqual(self.g, key_obj.g)
    self.assertEqual(self.x, key_obj.x)