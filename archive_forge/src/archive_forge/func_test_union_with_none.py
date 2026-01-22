import unittest
from traits.api import (
def test_union_with_none(self):

    class TestClass(HasTraits):
        int_or_none = Union(None, Int)
    TestClass(int_or_none=None)