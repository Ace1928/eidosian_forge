import unittest as pyunit
from twisted.python.reflect import namedAny
from twisted.trial import unittest
from twisted.trial.test import suppression
def test_suppressModule(self):
    """
        A suppression set on a module prevents warnings emitted by any test
        mewthods defined in that module which match the suppression from being
        emitted.
        """
    self.runTests(self._load(self.TestSuppression2, 'testSuppressModule'))
    warningsShown = self.flushWarnings([self.TestSuppression._emit])
    self.assertEqual(warningsShown[0]['message'], suppression.METHOD_WARNING_MSG)
    self.assertEqual(warningsShown[1]['message'], suppression.CLASS_WARNING_MSG)
    self.assertEqual(len(warningsShown), 2)