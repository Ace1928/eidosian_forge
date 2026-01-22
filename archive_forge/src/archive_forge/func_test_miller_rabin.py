import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import *
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math.Primality import (
def test_miller_rabin(self):
    for prime in self.primes:
        self.assertEqual(miller_rabin_test(prime, 3), PROBABLY_PRIME)
    for composite in self.composites:
        self.assertEqual(miller_rabin_test(composite, 3), COMPOSITE)
    self.assertRaises(ValueError, miller_rabin_test, -1, 3)