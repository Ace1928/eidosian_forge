from twisted.python.reflect import requireModule
from twisted.internet.address import IPv6Address
from twisted.internet.test.test_endpoints import deterministicResolvingReactor
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.trial import unittest
def makeTCPConnection(self, reactor: MemoryReactorClock) -> None:
    """
        Fake that connection was established for first connectTCP request made
        on C{reactor}.

        @param reactor: Reactor on which to fake the connection.
        @type  reactor: A reactor.
        """
    factory = reactor.tcpClients[0][2]
    connector = reactor.connectors[0]
    protocol = factory.buildProtocol(None)
    transport = StringTransport(peerAddress=connector.getDestination())
    protocol.makeConnection(transport)