import decimal
import sys
import unittest
from traits.api import Either, HasTraits, Int, CInt, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_respects_dunder_index(self):
    a = A()
    a.integral = IntegerLike()
    self.assertEqual(a.integral, 42)
    self.assertIs(type(a.integral), int)