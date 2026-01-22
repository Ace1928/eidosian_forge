import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromName__relative_empty_name(self):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('', unittest)
    error, test = self.check_deferred_error(loader, suite)
    expected = "has no attribute ''"
    self.assertIn(expected, error, 'missing error string in %r' % error)
    self.assertRaisesRegex(AttributeError, expected, getattr(test, ''))