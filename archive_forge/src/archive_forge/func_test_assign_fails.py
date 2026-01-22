import unittest
from traits.api import Constant, HasTraits, TraitError
def test_assign_fails(self):

    class TestClass(HasTraits):
        c_atr = Constant(5)
    with self.assertRaises(TraitError):
        TestClass(c_atr=5)
    with self.assertRaises(TraitError):
        del TestClass().c_atr