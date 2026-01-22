import unittest
from traits.api import BaseFloat, Either, Float, HasTraits, Str, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_compound_trait_float_conversion_fail(self):
    a = self.test_class()
    a.float_or_text = 'not a float'
    self.assertEqual(a.float_or_text, 'not a float')