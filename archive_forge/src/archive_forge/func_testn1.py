import unittest
from binascii import unhexlify as uh
from Cryptodome.Util.py3compat import *
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.Padding import pad, unpad
def testn1(self):
    self.assertRaises(ValueError, unpad, b('123456\x81'), 4, 'iso7816')
    self.assertRaises(ValueError, unpad, b(''), 4, 'iso7816')