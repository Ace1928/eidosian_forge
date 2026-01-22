import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_with_errors_in_addModuleCleanup_and_setUps(self):
    ordering = []
    module_blow_up = False
    class_blow_up = False
    method_blow_up = False

    class Module(object):

        @staticmethod
        def setUpModule():
            ordering.append('setUpModule')
            unittest.addModuleCleanup(cleanup, ordering, blowUp=True)
            if module_blow_up:
                raise CustomError('ModuleExc')

        @staticmethod
        def tearDownModule():
            ordering.append('tearDownModule')

    class TestableTest(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            ordering.append('setUpClass')
            if class_blow_up:
                raise CustomError('ClassExc')

        def setUp(self):
            ordering.append('setUp')
            if method_blow_up:
                raise CustomError('MethodExc')

        def testNothing(self):
            ordering.append('test')

        @classmethod
        def tearDownClass(cls):
            ordering.append('tearDownClass')
    TestableTest.__module__ = 'Module'
    sys.modules['Module'] = Module
    result = runTests(TestableTest)
    self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
    self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'setUp', 'test', 'tearDownClass', 'tearDownModule', 'cleanup_exc'])
    ordering = []
    module_blow_up = True
    class_blow_up = False
    method_blow_up = False
    result = runTests(TestableTest)
    self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: ModuleExc')
    self.assertEqual(result.errors[1][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
    self.assertEqual(ordering, ['setUpModule', 'cleanup_exc'])
    ordering = []
    module_blow_up = False
    class_blow_up = True
    method_blow_up = False
    result = runTests(TestableTest)
    self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: ClassExc')
    self.assertEqual(result.errors[1][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
    self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'tearDownModule', 'cleanup_exc'])
    ordering = []
    module_blow_up = False
    class_blow_up = False
    method_blow_up = True
    result = runTests(TestableTest)
    self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: MethodExc')
    self.assertEqual(result.errors[1][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
    self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'setUp', 'tearDownClass', 'tearDownModule', 'cleanup_exc'])