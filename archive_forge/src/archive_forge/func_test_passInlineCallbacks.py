from __future__ import annotations
import unittest as pyunit
from twisted.internet import defer
from twisted.python.failure import Failure
from twisted.trial import reporter, unittest, util
from twisted.trial.test import detests
def test_passInlineCallbacks(self) -> None:
    """
        The body of a L{defer.inlineCallbacks} decorated test gets run.
        """
    result = self.runTest('test_passInlineCallbacks')
    self.assertTrue(result.wasSuccessful())
    self.assertEqual(result.testsRun, 1)
    self.assertTrue(detests.DeferredTests.touched)