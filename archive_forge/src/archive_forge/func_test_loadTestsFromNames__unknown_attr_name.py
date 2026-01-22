import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromNames__unknown_attr_name(self):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(['unittest.loader.sdasfasfasdf', 'unittest.test.dummy'])
    error, test = self.check_deferred_error(loader, list(suite)[0])
    expected = "module 'unittest.loader' has no attribute 'sdasfasfasdf'"
    self.assertIn(expected, error, 'missing error string in %r' % error)
    self.assertRaisesRegex(AttributeError, expected, test.sdasfasfasdf)