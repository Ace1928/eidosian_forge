import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_with_errors_in_addClassCleanup_and_setUps(self):
    ordering = []
    class_blow_up = False
    method_blow_up = False

    class TestableTest(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            ordering.append('setUpClass')
            cls.addClassCleanup(cleanup, ordering, blowUp=True)
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
    result = runTests(TestableTest)
    self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
    self.assertEqual(ordering, ['setUpClass', 'setUp', 'test', 'tearDownClass', 'cleanup_exc'])
    ordering = []
    class_blow_up = True
    method_blow_up = False
    result = runTests(TestableTest)
    self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: ClassExc')
    self.assertEqual(result.errors[1][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
    self.assertEqual(ordering, ['setUpClass', 'cleanup_exc'])
    ordering = []
    class_blow_up = False
    method_blow_up = True
    result = runTests(TestableTest)
    self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: MethodExc')
    self.assertEqual(result.errors[1][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
    self.assertEqual(ordering, ['setUpClass', 'setUp', 'tearDownClass', 'cleanup_exc'])