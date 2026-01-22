import functools
import sys
import types
import warnings
import unittest
def test_sortTestMethodsUsing__default_value(self):
    loader = unittest.TestLoader()

    class Foo(unittest.TestCase):

        def test_2(self):
            pass

        def test_3(self):
            pass

        def test_1(self):
            pass
    test_names = ['test_2', 'test_3', 'test_1']
    self.assertEqual(loader.getTestCaseNames(Foo), sorted(test_names))