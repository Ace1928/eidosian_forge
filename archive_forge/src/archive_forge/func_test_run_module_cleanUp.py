import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_run_module_cleanUp(self):
    blowUp = True
    ordering = []

    class Module(object):

        @staticmethod
        def setUpModule():
            ordering.append('setUpModule')
            unittest.addModuleCleanup(cleanup, ordering)
            if blowUp:
                raise CustomError('setUpModule Exc')

        @staticmethod
        def tearDownModule():
            ordering.append('tearDownModule')

    class TestableTest(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            ordering.append('setUpClass')

        def testNothing(self):
            ordering.append('test')

        @classmethod
        def tearDownClass(cls):
            ordering.append('tearDownClass')
    TestableTest.__module__ = 'Module'
    sys.modules['Module'] = Module
    result = runTests(TestableTest)
    self.assertEqual(ordering, ['setUpModule', 'cleanup_good'])
    self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: setUpModule Exc')
    ordering = []
    blowUp = False
    runTests(TestableTest)
    self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'test', 'tearDownClass', 'tearDownModule', 'cleanup_good'])
    self.assertEqual(unittest.case._module_cleanups, [])