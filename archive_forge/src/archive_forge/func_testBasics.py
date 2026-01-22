import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def testBasics(self):
    """
        Test that a normal deferredGenerator works.  Tests yielding a
        deferred which callbacks, as well as a deferred errbacks. Also
        ensures returning a final value works.
        """
    return self._genBasics().addCallback(self.assertEqual, 'WOOSH')