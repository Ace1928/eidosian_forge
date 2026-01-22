import functools
import sys
import types
import warnings
import unittest
def test_suiteClass__loadTestsFromNames(self):
    m = types.ModuleType('m')

    class Foo(unittest.TestCase):

        def test_1(self):
            pass

        def test_2(self):
            pass

        def foo_bar(self):
            pass
    m.Foo = Foo
    tests = [[Foo('test_1'), Foo('test_2')]]
    loader = unittest.TestLoader()
    loader.suiteClass = list
    self.assertEqual(loader.loadTestsFromNames(['Foo'], m), tests)