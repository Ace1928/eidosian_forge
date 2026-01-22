import unittest
from traits.testing.optional_dependencies import optional_import
def test_import_succeeds(self):
    module = optional_import('itertools')
    self.assertEqual(module.__name__, 'itertools')