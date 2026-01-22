import io
import sys
import unittest
def test_error_in_setup_module(self):

    class Module(object):
        moduleSetup = 0
        moduleTornDown = 0

        @staticmethod
        def setUpModule():
            Module.moduleSetup += 1
            raise TypeError('foo')

        @staticmethod
        def tearDownModule():
            Module.moduleTornDown += 1

    class Test(unittest.TestCase):
        classSetUp = False
        classTornDown = False

        @classmethod
        def setUpClass(cls):
            Test.classSetUp = True

        @classmethod
        def tearDownClass(cls):
            Test.classTornDown = True

        def test_one(self):
            pass

        def test_two(self):
            pass

    class Test2(unittest.TestCase):

        def test_one(self):
            pass

        def test_two(self):
            pass
    Test.__module__ = 'Module'
    Test2.__module__ = 'Module'
    sys.modules['Module'] = Module
    result = self.runTests(Test, Test2)
    self.assertEqual(Module.moduleSetup, 1)
    self.assertEqual(Module.moduleTornDown, 0)
    self.assertEqual(result.testsRun, 0)
    self.assertFalse(Test.classSetUp)
    self.assertFalse(Test.classTornDown)
    self.assertEqual(len(result.errors), 1)
    error, _ = result.errors[0]
    self.assertEqual(str(error), 'setUpModule (Module)')