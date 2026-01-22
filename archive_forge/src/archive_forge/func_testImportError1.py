import unittest
import re
from Cryptodome.PublicKey import DSA
from Cryptodome.SelfTest.st_common import *
from Cryptodome.Util.py3compat import *
from binascii import unhexlify
def testImportError1(self):
    self.assertRaises(ValueError, DSA.importKey, self.der_pkcs8_encrypted, 'wrongpwd')