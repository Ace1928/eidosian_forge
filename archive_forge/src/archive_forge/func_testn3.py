import unittest
from binascii import unhexlify as uh
from Cryptodome.Util.py3compat import *
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.Padding import pad, unpad
def testn3(self):
    self.assertRaises(ValueError, unpad, b('123456\x02'), 4)
    self.assertRaises(ValueError, unpad, b('123456\x00'), 4)
    self.assertRaises(ValueError, unpad, b('123456\x05\x05\x05\x05\x05'), 4)