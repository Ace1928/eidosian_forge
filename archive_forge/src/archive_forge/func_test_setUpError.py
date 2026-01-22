from __future__ import annotations
import sys
import traceback
import unittest as pyunit
from unittest import skipIf
from zope.interface import implementer
from twisted.python.failure import Failure
from twisted.trial.itrial import IReporter, ITestCase
from twisted.trial.test import pyunitcases
from twisted.trial.unittest import PyUnitResultAdapter, SynchronousTestCase
def test_setUpError(self) -> None:

    class ErrorTest(SynchronousTestCase):
        ran = False

        def setUp(self) -> None:
            1 / 0

        def test_foo(s) -> None:
            s.ran = True
    test = ErrorTest('test_foo')
    result = pyunit.TestResult()
    test.run(result)
    self.assertFalse(test.ran)
    self.assertEqual(1, result.testsRun)
    self.assertEqual(1, len(result.errors))
    self.assertFalse(result.wasSuccessful())