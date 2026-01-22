import decimal
import sys
import unittest
from traits.api import Either, HasTraits, Int, CInt, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_cint_conversion_of_subclasses(self):
    a = A()
    a.convertible = True
    self.assertIs(type(a.convertible), int)
    self.assertEqual(a.convertible, 1)
    a.convertible_or_none = True
    self.assertIs(type(a.convertible_or_none), int)
    self.assertEqual(a.convertible_or_none, 1)