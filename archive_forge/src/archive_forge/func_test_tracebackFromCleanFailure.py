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
def test_tracebackFromCleanFailure(self) -> None:
    """
        Errors added through the L{PyUnitResultAdapter} have the same
        traceback information as if there were no adapter at all, even
        if the Failure that held the information has been cleaned.
        """
    try:
        1 / 0
    except ZeroDivisionError:
        exc_info = sys.exc_info()
        f = Failure()
    f.cleanFailure()
    pyresult = pyunit.TestResult()
    result = PyUnitResultAdapter(pyresult)
    result.addError(self, f)
    tback = ''.join(traceback.format_exception(*exc_info))
    self.assertEqual(pyresult.errors[0][1].endswith('ZeroDivisionError: division by zero\n'), tback.endswith('ZeroDivisionError: division by zero\n'))