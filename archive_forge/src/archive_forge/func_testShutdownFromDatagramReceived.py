import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def testShutdownFromDatagramReceived(self):
    """Test reactor shutdown while in a recvfrom() loop"""
    finished = defer.Deferred()
    pr = self.server.packetReceived = defer.Deferred()

    def pktRece(ignored):
        self.server.transport.connectionLost()
        reactor.callLater(0, finished.callback, None)
    pr.addCallback(pktRece)

    def flushErrors(ignored):
        self.flushLoggedErrors()
    finished.addCallback(flushErrors)
    self.server.transport.write(b'\x00' * 64, ('127.0.0.1', self.server.transport.getHost().port))
    return finished