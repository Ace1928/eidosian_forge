import unittest as pyunit
from twisted.python.reflect import namedAny
from twisted.trial import unittest
from twisted.trial.test import suppression
def test_setUpSuppression(self):
    """
        Suppressions defined by the test method being run are applied to any
        warnings emitted while running the C{setUp} fixture.
        """
    self.runTests(self._load(self.TestSetUpSuppression, 'testSuppressMethod'))
    warningsShown = self.flushWarnings([self.TestSetUpSuppression._emit])
    self._assertWarnings(warningsShown, [suppression.CLASS_WARNING_MSG, suppression.MODULE_WARNING_MSG, suppression.CLASS_WARNING_MSG, suppression.MODULE_WARNING_MSG])