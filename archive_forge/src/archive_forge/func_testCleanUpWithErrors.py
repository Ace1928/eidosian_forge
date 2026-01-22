import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def testCleanUpWithErrors(self):

    class TestableTest(unittest.TestCase):

        def testNothing(self):
            pass
    test = TestableTest('testNothing')
    result = unittest.TestResult()
    outcome = test._outcome = _Outcome(result=result)
    CleanUpExc = CustomError('foo')
    exc2 = CustomError('bar')

    def cleanup1():
        raise CleanUpExc

    def cleanup2():
        raise exc2
    test.addCleanup(cleanup1)
    test.addCleanup(cleanup2)
    self.assertFalse(test.doCleanups())
    self.assertFalse(outcome.success)
    (_, msg2), (_, msg1) = result.errors
    self.assertIn('in cleanup1', msg1)
    self.assertIn('raise CleanUpExc', msg1)
    self.assertIn(f'{CustomErrorRepr}: foo', msg1)
    self.assertIn('in cleanup2', msg2)
    self.assertIn('raise exc2', msg2)
    self.assertIn(f'{CustomErrorRepr}: bar', msg2)