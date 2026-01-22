import unittest
from binascii import unhexlify, hexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.strxor import strxor, strxor_c
def test_output_overlapping_bytearray(self):
    """Verify result can be stored in overlapping memory"""
    term1 = bytearray(unhexlify(b'ff339a83e5cd4cdf5649'))
    expected_xor = unhexlify(b'be72dbc2a48c0d9e1708')
    result = strxor_c(term1, 65, output=term1)
    self.assertEqual(result, None)
    self.assertEqual(term1, expected_xor)