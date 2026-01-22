import unittest as pyunit
from twisted.python.reflect import namedAny
from twisted.trial import unittest
from twisted.trial.test import suppression
def test_tearDownSuppression(self):
    """
        Suppressions defined by the test method being run are applied to any
        warnings emitted while running the C{tearDown} fixture.
        """
    self.runTests(self._load(self.TestTearDownSuppression, 'testSuppressMethod'))
    warningsShown = self.flushWarnings([self.TestTearDownSuppression._emit])
    self._assertWarnings(warningsShown, [suppression.CLASS_WARNING_MSG, suppression.MODULE_WARNING_MSG, suppression.CLASS_WARNING_MSG, suppression.MODULE_WARNING_MSG])