import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def test_expected_nr_elements(self):
    der_bin = DerSequence([1, 2, 3]).encode()
    DerSequence().decode(der_bin, nr_elements=3)
    DerSequence().decode(der_bin, nr_elements=(2, 3))
    self.assertRaises(ValueError, DerSequence().decode, der_bin, nr_elements=1)
    self.assertRaises(ValueError, DerSequence().decode, der_bin, nr_elements=(4, 5))