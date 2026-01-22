import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
def test_active(self):
    """
        L{IDelayedCall.active} returns False once the call has run.
        """
    dcall = reactor.callLater(0.01, self.deferred.callback, True)
    self.assertTrue(dcall.active())

    def checkDeferredCall(success):
        self.assertFalse(dcall.active())
        return success
    self.deferred.addCallback(checkDeferredCall)
    return self.deferred