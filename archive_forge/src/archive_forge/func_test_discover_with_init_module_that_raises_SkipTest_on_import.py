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
def test_discover_with_init_module_that_raises_SkipTest_on_import(self):
    if not unittest.BaseTestSuite._cleanup:
        raise unittest.SkipTest('Suite cleanup is disabled')
    vfs = {abspath('/foo'): ['my_package'], abspath('/foo/my_package'): ['__init__.py', 'test_module.py']}
    self.setup_import_issue_package_tests(vfs)
    import_calls = []

    def _get_module_from_name(name):
        import_calls.append(name)
        raise unittest.SkipTest('skipperoo')
    loader = unittest.TestLoader()
    loader._get_module_from_name = _get_module_from_name
    suite = loader.discover(abspath('/foo'))
    self.assertIn(abspath('/foo'), sys.path)
    self.assertEqual(suite.countTestCases(), 1)
    result = unittest.TestResult()
    suite.run(result)
    self.assertEqual(len(result.skipped), 1)
    self.assertEqual(result.testsRun, 1)
    self.assertEqual(import_calls, ['my_package'])
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        pickle.loads(pickle.dumps(suite, proto))