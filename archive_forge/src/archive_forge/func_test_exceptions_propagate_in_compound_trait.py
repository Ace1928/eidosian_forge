import unittest
from traits.api import BaseFloat, Either, Float, HasTraits, Str, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_exceptions_propagate_in_compound_trait(self):
    a = self.test_class()
    with self.assertRaises(ZeroDivisionError):
        a.value_or_none = BadFloat()