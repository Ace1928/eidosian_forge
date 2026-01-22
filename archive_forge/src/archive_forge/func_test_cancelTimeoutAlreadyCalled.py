import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_cancelTimeoutAlreadyCalled(self):
    """
        Does nothing if no timeout is already reached.
        """
    sut, clock = self.getProtocolAndClock()
    wrappedProto = sut.wrappedProtocol
    sut.setTimeout(3)
    clock.advance(3)
    self.assertTrue(wrappedProto.disconnected)
    sut.cancelTimeout()