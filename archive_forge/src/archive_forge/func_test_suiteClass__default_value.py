import functools
import sys
import types
import warnings
import unittest
def test_suiteClass__default_value(self):
    loader = unittest.TestLoader()
    self.assertIs(loader.suiteClass, unittest.TestSuite)