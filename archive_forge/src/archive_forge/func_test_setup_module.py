import io
import sys
import unittest
def test_setup_module(self):

    class Module(object):
        moduleSetup = 0

        @staticmethod
        def setUpModule():
            Module.moduleSetup += 1

    class Test(unittest.TestCase):

        def test_one(self):
            pass

        def test_two(self):
            pass
    Test.__module__ = 'Module'
    sys.modules['Module'] = Module
    result = self.runTests(Test)
    self.assertEqual(Module.moduleSetup, 1)
    self.assertEqual(result.testsRun, 2)
    self.assertEqual(len(result.errors), 0)