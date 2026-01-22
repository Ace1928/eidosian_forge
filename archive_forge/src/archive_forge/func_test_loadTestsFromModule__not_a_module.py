import functools
import sys
import types
import warnings
import unittest
def test_loadTestsFromModule__not_a_module(self):

    class MyTestCase(unittest.TestCase):

        def test(self):
            pass

    class NotAModule(object):
        test_2 = MyTestCase
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(NotAModule)
    reference = [unittest.TestSuite([MyTestCase('test')])]
    self.assertEqual(list(suite), reference)