import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def cbPacketReceived(packet):
    """
                A packet arrived.  Cancel the timeout for it, record it, and
                maybe finish the test.
                """
    timeoutCall.cancel()
    succeededAttempts.append(packet)
    if len(succeededAttempts) == 2:
        reactor.callLater(0, finalDeferred.callback, None)
    else:
        makeAttempt()