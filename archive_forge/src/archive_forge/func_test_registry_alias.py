import os
import sys
from breezy import branch, osutils, registry, tests
def test_registry_alias(self):
    a_registry = registry.Registry()
    a_registry.register('one', 1, info='string info')
    a_registry.register_alias('two', 'one')
    a_registry.register_alias('three', 'one', info='own info')
    self.assertEqual(a_registry.get('one'), a_registry.get('two'))
    self.assertEqual(a_registry.get_help('one'), a_registry.get_help('two'))
    self.assertEqual(a_registry.get_info('one'), a_registry.get_info('two'))
    self.assertEqual('own info', a_registry.get_info('three'))
    self.assertEqual({'two': 'one', 'three': 'one'}, a_registry.aliases())
    self.assertEqual({'one': ['three', 'two']}, {k: sorted(v) for k, v in a_registry.alias_map().items()})