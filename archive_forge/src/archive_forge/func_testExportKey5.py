import unittest
import re
from Cryptodome.PublicKey import DSA
from Cryptodome.SelfTest.st_common import *
from Cryptodome.Util.py3compat import *
from binascii import unhexlify
def testExportKey5(self):
    tup = (self.y, self.g, self.p, self.q, self.x)
    key = DSA.construct(tup)
    encoded = key.export_key('DER')
    self.assertEqual(self.der_pkcs8, encoded)
    encoded = key.export_key('DER', pkcs8=True)
    self.assertEqual(self.der_pkcs8, encoded)