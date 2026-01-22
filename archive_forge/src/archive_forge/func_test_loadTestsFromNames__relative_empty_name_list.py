import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromNames__relative_empty_name_list(self):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames([], unittest)
    self.assertIsInstance(suite, loader.suiteClass)
    self.assertEqual(list(suite), [])