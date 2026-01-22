import unittest
from traits.api import BaseFloat, Either, Float, HasTraits, Str, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
@requires_numpy
def test_accepts_numpy_floats(self):
    test_values = [numpy.float64(2.3), numpy.float32(3.7), numpy.float16(1.28)]
    a = self.test_class()
    for test_value in test_values:
        a.value = test_value
        self.assertIs(type(a.value), float)
        self.assertEqual(a.value, test_value)
        a.value_or_none = test_value
        self.assertIs(type(a.value_or_none), float)
        self.assertEqual(a.value_or_none, test_value)