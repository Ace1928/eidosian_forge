import unittest
from traits.api import (
def test_list_trait_instances(self):

    class TestClass(HasTraits):
        float_or_str_obj = Union(Instance(Float), Instance(Str))
    TestClass(float_or_str_obj=Float(3.5))
    TestClass(float_or_str_obj=Str('3.5'))
    with self.assertRaises(TraitError):
        TestClass(float_or_str_obj=Float)
    with self.assertRaises(TraitError):
        TestClass(float_or_str_obj=3.5)