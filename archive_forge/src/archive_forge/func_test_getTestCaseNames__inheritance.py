import functools
import sys
import types
import warnings
import unittest
def test_getTestCaseNames__inheritance(self):

    class TestP(unittest.TestCase):

        def test_1(self):
            pass

        def test_2(self):
            pass

        def foobar(self):
            pass

    class TestC(TestP):

        def test_1(self):
            pass

        def test_3(self):
            pass
    loader = unittest.TestLoader()
    names = ['test_1', 'test_2', 'test_3']
    self.assertEqual(loader.getTestCaseNames(TestC), names)