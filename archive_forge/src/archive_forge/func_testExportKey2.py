import unittest
import re
from Cryptodome.PublicKey import DSA
from Cryptodome.SelfTest.st_common import *
from Cryptodome.Util.py3compat import *
from binascii import unhexlify
def testExportKey2(self):
    tup = (self.y, self.g, self.p, self.q)
    key = DSA.construct(tup)
    encoded = key.export_key('PEM')
    self.assertEqual(self.pem_public, encoded)