import unittest
from traits.api import (
def test_union_user_defined_type(self):

    class TestClass(HasTraits):
        type_value = Union(CustomStrType, Int)
    TestClass(type_value='new string')