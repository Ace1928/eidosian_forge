import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def test_waitForDeferredDeprecated(self):
    """
        L{waitForDeferred} is deprecated.
        """
    d = Deferred()
    waitForDeferred(d)
    warnings = self.flushWarnings([self.test_waitForDeferredDeprecated])
    self.assertEqual(len(warnings), 1)
    self.assertEqual(warnings[0]['category'], DeprecationWarning)
    self.assertEqual(warnings[0]['message'], 'twisted.internet.defer.waitForDeferred was deprecated in Twisted 15.0.0; please use twisted.internet.defer.inlineCallbacks instead')