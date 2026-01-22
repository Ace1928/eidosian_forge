import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def testBuggy(self):
    """
        Ensure that a buggy generator properly signals a Failure
        condition on result deferred.
        """
    return self.assertFailure(self._genBuggy(), ZeroDivisionError)