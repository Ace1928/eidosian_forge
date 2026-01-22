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
def test_trialSkip(self) -> None:
    """
        Skips using trial's skipping functionality are reported as skips in
        the L{pyunit.TestResult}.
        """

    class SkipTest(SynchronousTestCase):

        @skipIf(True, "Let's skip!")
        def test_skip(self) -> None:
            1 / 0
    test = SkipTest('test_skip')
    result = pyunit.TestResult()
    test.run(result)
    self.assertEqual(result.skipped, [(test, "Let's skip!")])