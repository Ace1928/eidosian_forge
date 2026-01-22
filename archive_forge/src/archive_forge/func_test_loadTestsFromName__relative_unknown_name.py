import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromName__relative_unknown_name(self):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('sdasfasfasdf', unittest)
    expected = "module 'unittest' has no attribute 'sdasfasfasdf'"
    error, test = self.check_deferred_error(loader, suite)
    self.assertIn(expected, error, 'missing error string in %r' % error)
    self.assertRaisesRegex(AttributeError, expected, test.sdasfasfasdf)