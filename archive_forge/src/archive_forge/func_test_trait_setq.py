import unittest
from traits.api import HasTraits, Str, Int
from traits.testing.unittest_tools import UnittestTools
def test_trait_setq(self):
    obj = TraitsObject()
    obj.string = 'foo'
    with self.assertTraitDoesNotChange(obj, 'string'):
        obj.trait_setq(string='bar')
    self.assertEqual(obj.string, 'bar')