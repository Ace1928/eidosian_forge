from __future__ import annotations
import unittest as pyunit
from twisted.internet import defer
from twisted.python.failure import Failure
from twisted.trial import reporter, unittest, util
from twisted.trial.test import detests
def test_errorPropagation(self) -> None:
    result = self.runTest('test_errorPropagation')
    self.assertFalse(result.wasSuccessful())
    self.assertEqual(result.testsRun, 1)
    assert detests.TimeoutTests.timedOut is not None
    self._wasTimeout(detests.TimeoutTests.timedOut)