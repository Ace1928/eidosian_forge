import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def test_multicast(self):
    """
        Test that a multicast group can be joined and messages sent to and
        received from it.
        """
    c = Server()
    p = reactor.listenMulticast(0, c)
    addr = self.server.transport.getHost()
    joined = self.server.transport.joinGroup('225.0.0.250')

    def cbJoined(ignored):
        d = self.server.packetReceived = Deferred()
        c.transport.write(b'hello world', ('225.0.0.250', addr.port))
        return d
    joined.addCallback(cbJoined)

    def cbPacket(ignored):
        self.assertEqual(self.server.packets[0][0], b'hello world')
    joined.addCallback(cbPacket)

    def cleanup(passthrough):
        result = maybeDeferred(p.stopListening)
        result.addCallback(lambda ign: passthrough)
        return result
    joined.addCallback(cleanup)
    return joined