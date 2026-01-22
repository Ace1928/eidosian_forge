from __future__ import annotations
from io import StringIO
from twisted.trial import reporter, runner, unittest
def test_setUpOK(self) -> None:
    self.assertEqual(0, InterruptInTearDownTests.testsRun)
    self.assertEqual(2, self.suite.countTestCases())
    self.assertEqual(0, self.reporter.testsRun)
    self.assertFalse(self.reporter.shouldStop)