import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromModule__no_TestCase_instances(self):
    m = types.ModuleType('m')
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(m)
    self.assertIsInstance(suite, loader.suiteClass)
    self.assertEqual(list(suite), [])