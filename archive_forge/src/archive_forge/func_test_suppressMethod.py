import unittest as pyunit
from twisted.python.reflect import namedAny
from twisted.trial import unittest
from twisted.trial.test import suppression
def test_suppressMethod(self):
    """
        A suppression set on a test method prevents warnings emitted by that
        test method which the suppression matches from being emitted.
        """
    self.runTests(self._load(self.TestSuppression, 'testSuppressMethod'))
    warningsShown = self.flushWarnings([self.TestSuppression._emit])
    self._assertWarnings(warningsShown, [suppression.CLASS_WARNING_MSG, suppression.MODULE_WARNING_MSG])