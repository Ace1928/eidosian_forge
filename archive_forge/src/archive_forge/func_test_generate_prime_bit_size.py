import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math.Primality import (
def test_generate_prime_bit_size(self):
    p = generate_probable_prime(exact_bits=512)
    self.assertEqual(p.size_in_bits(), 512)