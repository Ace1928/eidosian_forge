import unittest
from traits.api import (
def test_union_user_defined_class(self):

    class TestClass(HasTraits):
        obj = Union(Instance(CustomClass), Int)
    TestClass(obj=CustomClass(value=5))
    TestClass(obj=5)
    with self.assertRaises(TraitError):
        TestClass(obj=CustomClass)