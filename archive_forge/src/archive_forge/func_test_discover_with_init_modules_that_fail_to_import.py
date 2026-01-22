import os.path
from os.path import abspath
import re
import sys
import types
import pickle
from test import support
from test.support import import_helper
import unittest
import unittest.mock
import unittest.test
def test_discover_with_init_modules_that_fail_to_import(self):
    vfs = {abspath('/foo'): ['my_package'], abspath('/foo/my_package'): ['__init__.py', 'test_module.py']}
    self.setup_import_issue_package_tests(vfs)
    import_calls = []

    def _get_module_from_name(name):
        import_calls.append(name)
        raise ImportError('Cannot import Name')
    loader = unittest.TestLoader()
    loader._get_module_from_name = _get_module_from_name
    suite = loader.discover(abspath('/foo'))
    self.assertIn(abspath('/foo'), sys.path)
    self.assertEqual(suite.countTestCases(), 1)
    self.assertNotEqual([], loader.errors)
    self.assertEqual(1, len(loader.errors))
    error = loader.errors[0]
    self.assertTrue('Failed to import test module: my_package' in error, 'missing error string in %r' % error)
    test = list(list(suite)[0])[0]
    with self.assertRaises(ImportError):
        test.my_package()
    self.assertEqual(import_calls, ['my_package'])
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        pickle.loads(pickle.dumps(test, proto))