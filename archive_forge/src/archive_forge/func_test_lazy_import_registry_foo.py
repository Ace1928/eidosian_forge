import os
import sys
from breezy import branch, osutils, registry, tests
def test_lazy_import_registry_foo(self):
    a_registry = registry.Registry()
    a_registry.register_lazy('foo', 'breezy.branch', 'Branch')
    a_registry.register_lazy('bar', 'breezy.branch', 'Branch.hooks')
    self.assertEqual(branch.Branch, a_registry.get('foo'))
    self.assertEqual(branch.Branch.hooks, a_registry.get('bar'))