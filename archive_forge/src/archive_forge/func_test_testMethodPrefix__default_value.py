import functools
import sys
import types
import warnings
import unittest
def test_testMethodPrefix__default_value(self):
    loader = unittest.TestLoader()
    self.assertEqual(loader.testMethodPrefix, 'test')