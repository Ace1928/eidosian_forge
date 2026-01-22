import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def test_bindError(self):
    """
        A L{CannotListenError} exception is raised when attempting to bind a
        second protocol instance to an already bound port
        """
    server = Server()
    d = server.startedDeferred = defer.Deferred()
    port = reactor.listenUDP(0, server, interface='127.0.0.1')

    def cbStarted(ignored):
        self.assertEqual(port.getHost(), server.transport.getHost())
        server2 = Server()
        self.assertRaises(error.CannotListenError, reactor.listenUDP, port.getHost().port, server2, interface='127.0.0.1')
    d.addCallback(cbStarted)

    def cbFinished(ignored):
        return port.stopListening()
    d.addCallback(cbFinished)
    return d