import unittest
from traits.api import ArrayOrNone, ComparisonMode, HasTraits, TraitError
from traits.testing.unittest_tools import UnittestTools
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_setting_array_from_array(self):
    foo = Foo()
    test_array = numpy.arange(5)
    foo.maybe_array = test_array
    output_array = foo.maybe_array
    self.assertIsInstance(output_array, numpy.ndarray)
    self.assertEqual(output_array.dtype, test_array.dtype)
    self.assertEqual(output_array.shape, test_array.shape)
    self.assertTrue((output_array == test_array).all())