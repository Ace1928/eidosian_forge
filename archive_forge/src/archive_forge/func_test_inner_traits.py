import unittest
from traits.api import (
def test_inner_traits(self):

    class TestClass(HasTraits):
        atr = Union(Float, Int, Str)
    obj = TestClass()
    t1, t2, t3 = obj.trait('atr').inner_traits
    self.assertEqual(type(t1.trait_type), Float)
    self.assertEqual(type(t2.trait_type), Int)
    self.assertEqual(type(t3.trait_type), Str)