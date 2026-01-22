import unittest
from traits.api import ArrayOrNone, ComparisonMode, HasTraits, TraitError
from traits.testing.unittest_tools import UnittestTools
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_setting_array_from_list(self):
    foo = Foo()
    test_list = [5, 6, 7, 8, 9]
    foo.maybe_array = test_list
    output_array = foo.maybe_array
    self.assertIsInstance(output_array, numpy.ndarray)
    self.assertEqual(output_array.dtype, numpy.dtype(int))
    self.assertEqual(output_array.shape, (5,))
    self.assertTrue((output_array == test_list).all())