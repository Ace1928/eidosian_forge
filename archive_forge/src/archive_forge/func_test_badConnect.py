import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def test_badConnect(self):
    """
        A call to the transport's connect method fails with an
        L{InvalidAddressError} when a non-IP address is passed as the host
        value.

        A call to a transport's connect method fails with a L{RuntimeError}
        when the transport is already connected.
        """
    client = GoodClient()
    port = reactor.listenUDP(0, client, interface='127.0.0.1')
    self.assertRaises(error.InvalidAddressError, client.transport.connect, 'localhost', 80)
    client.transport.connect('127.0.0.1', 80)
    self.assertRaises(RuntimeError, client.transport.connect, '127.0.0.1', 80)
    return port.stopListening()