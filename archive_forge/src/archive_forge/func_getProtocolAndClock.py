import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def getProtocolAndClock(self):
    """
        Helper to set up an already connected protocol to be tested.

        @return: A new protocol with its attached clock.
        @rtype: Tuple of (L{policies.TimeoutProtocol}, L{task.Clock})
        """
    clock = task.Clock()
    wrappedFactory = protocol.ServerFactory()
    wrappedFactory.protocol = SimpleProtocol
    factory = TestableTimeoutFactory(clock, wrappedFactory, None)
    proto = factory.buildProtocol(address.IPv4Address('TCP', '127.0.0.1', 12345))
    transport = StringTransportWithDisconnection()
    transport.protocol = proto
    proto.makeConnection(transport)
    return (proto, clock)