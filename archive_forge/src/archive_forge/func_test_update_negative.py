import unittest
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TupleHash128, TupleHash256
def test_update_negative(self):
    h = self.new()
    self.assertRaises(TypeError, h.update, u'string')
    self.assertRaises(TypeError, h.update, None)
    self.assertRaises(TypeError, h.update, (b'STRING1', b'STRING2'))