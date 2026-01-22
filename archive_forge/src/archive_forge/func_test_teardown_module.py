import io
import sys
import unittest
def test_teardown_module(self):

    class Module(object):
        moduleTornDown = 0

        @staticmethod
        def tearDownModule():
            Module.moduleTornDown += 1

    class Test(unittest.TestCase):

        def test_one(self):
            pass

        def test_two(self):
            pass
    Test.__module__ = 'Module'
    sys.modules['Module'] = Module
    result = self.runTests(Test)
    self.assertEqual(Module.moduleTornDown, 1)
    self.assertEqual(result.testsRun, 2)
    self.assertEqual(len(result.errors), 0)