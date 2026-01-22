import unittest
from traits.api import Bool, Dict, HasTraits, Int, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_does_not_accept_int_or_float(self):
    a = A()
    bad_values = [-1, 'a string', 1.0]
    for bad_value in bad_values:
        with self.assertRaises(TraitError):
            a.foo = bad_value
    self.assertEqual(type(a.foo), bool)
    self.assertFalse(a.foo)