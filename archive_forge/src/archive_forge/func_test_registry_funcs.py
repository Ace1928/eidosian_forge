import os
import sys
from breezy import branch, osutils, registry, tests
def test_registry_funcs(self):
    a_registry = registry.Registry()
    self.register_stuff(a_registry)
    self.assertTrue('one' in a_registry)
    a_registry.remove('one')
    self.assertFalse('one' in a_registry)
    self.assertRaises(KeyError, a_registry.get, 'one')
    a_registry.register('one', 'one')
    self.assertEqual(['five', 'four', 'one', 'two'], sorted(a_registry.keys()))
    self.assertEqual([('five', 5), ('four', 4), ('one', 'one'), ('two', 2)], sorted(a_registry.iteritems()))