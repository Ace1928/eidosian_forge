import decimal
import sys
import unittest
from traits.api import Either, HasTraits, Int, CInt, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_rejects_floating_point_types(self):
    a = A()
    with self.assertRaises(TraitError):
        a.integral = 23.0
    with self.assertRaises(TraitError):
        a.integral = decimal.Decimal(23)