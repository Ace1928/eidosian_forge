import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_run_multiple_module_cleanUp(self):
    blowUp = True
    blowUp2 = False
    ordering = []

    class Module1(object):

        @staticmethod
        def setUpModule():
            ordering.append('setUpModule')
            unittest.addModuleCleanup(cleanup, ordering)
            if blowUp:
                raise CustomError()

        @staticmethod
        def tearDownModule():
            ordering.append('tearDownModule')

    class Module2(object):

        @staticmethod
        def setUpModule():
            ordering.append('setUpModule2')
            unittest.addModuleCleanup(cleanup, ordering)
            if blowUp2:
                raise CustomError()

        @staticmethod
        def tearDownModule():
            ordering.append('tearDownModule2')

    class TestableTest(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            ordering.append('setUpClass')

        def testNothing(self):
            ordering.append('test')

        @classmethod
        def tearDownClass(cls):
            ordering.append('tearDownClass')

    class TestableTest2(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            ordering.append('setUpClass2')

        def testNothing(self):
            ordering.append('test2')

        @classmethod
        def tearDownClass(cls):
            ordering.append('tearDownClass2')
    TestableTest.__module__ = 'Module1'
    sys.modules['Module1'] = Module1
    TestableTest2.__module__ = 'Module2'
    sys.modules['Module2'] = Module2
    runTests(TestableTest, TestableTest2)
    self.assertEqual(ordering, ['setUpModule', 'cleanup_good', 'setUpModule2', 'setUpClass2', 'test2', 'tearDownClass2', 'tearDownModule2', 'cleanup_good'])
    ordering = []
    blowUp = False
    blowUp2 = True
    runTests(TestableTest, TestableTest2)
    self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'test', 'tearDownClass', 'tearDownModule', 'cleanup_good', 'setUpModule2', 'cleanup_good'])
    ordering = []
    blowUp = False
    blowUp2 = False
    runTests(TestableTest, TestableTest2)
    self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'test', 'tearDownClass', 'tearDownModule', 'cleanup_good', 'setUpModule2', 'setUpClass2', 'test2', 'tearDownClass2', 'tearDownModule2', 'cleanup_good'])
    self.assertEqual(unittest.case._module_cleanups, [])