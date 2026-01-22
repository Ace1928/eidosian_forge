import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def testReturnValue(self):
    """Ensure that returnValue works."""

    def _return():
        yield 5
        returnValue(6)
    _return = inlineCallbacks(_return)
    return _return().addCallback(self.assertEqual, 6)