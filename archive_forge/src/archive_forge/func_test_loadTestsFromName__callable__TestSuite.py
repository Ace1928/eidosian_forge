import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromName__callable__TestSuite(self):
    m = types.ModuleType('m')
    testcase_1 = unittest.FunctionTestCase(lambda: None)
    testcase_2 = unittest.FunctionTestCase(lambda: None)

    def return_TestSuite():
        return unittest.TestSuite([testcase_1, testcase_2])
    m.return_TestSuite = return_TestSuite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('return_TestSuite', m)
    self.assertIsInstance(suite, loader.suiteClass)
    self.assertEqual(list(suite), [testcase_1, testcase_2])