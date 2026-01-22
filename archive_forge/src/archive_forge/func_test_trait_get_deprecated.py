import unittest
from traits.api import HasTraits, Str, Int
from traits.testing.unittest_tools import UnittestTools
def test_trait_get_deprecated(self):
    obj = TraitsObject()
    obj.string = 'foo'
    obj.integer = 1
    with self.assertNotDeprecated():
        values = obj.trait_get('integer')
    self.assertEqual(values, {'integer': 1})
    with self.assertDeprecated():
        values = obj.get('string')
    self.assertEqual(values, {'string': 'foo'})