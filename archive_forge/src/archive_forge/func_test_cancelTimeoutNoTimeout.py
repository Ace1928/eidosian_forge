import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_cancelTimeoutNoTimeout(self):
    """
        Does nothing if no timeout is already set.
        """
    sut, clock = self.getProtocolAndClock()
    self.assertIsNone(sut.timeoutCall)
    sut.cancelTimeout()
    self.assertFalse(sut.wrappedProtocol.disconnected)