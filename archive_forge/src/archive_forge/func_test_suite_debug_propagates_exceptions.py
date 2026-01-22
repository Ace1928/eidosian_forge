import io
import sys
import unittest
def test_suite_debug_propagates_exceptions(self):

    class Module(object):

        @staticmethod
        def setUpModule():
            if phase == 0:
                raise Exception('setUpModule')

        @staticmethod
        def tearDownModule():
            if phase == 1:
                raise Exception('tearDownModule')

    class Test(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            if phase == 2:
                raise Exception('setUpClass')

        @classmethod
        def tearDownClass(cls):
            if phase == 3:
                raise Exception('tearDownClass')

        def test_something(self):
            if phase == 4:
                raise Exception('test_something')
    Test.__module__ = 'Module'
    sys.modules['Module'] = Module
    messages = ('setUpModule', 'tearDownModule', 'setUpClass', 'tearDownClass', 'test_something')
    for phase, msg in enumerate(messages):
        _suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test)
        suite = unittest.TestSuite([_suite])
        with self.assertRaisesRegex(Exception, msg):
            suite.debug()