import unittest
from traits.api import (
def test_malformed_declaration(self):
    with self.assertRaises(ValueError):

        class TestClass(HasTraits):
            a = Union(int, Float)
        TestClass(a=2.4)
    with self.assertRaises(ValueError):

        class TestClass(HasTraits):
            a = Union([1, 2], Float)
        TestClass(a=2.4)