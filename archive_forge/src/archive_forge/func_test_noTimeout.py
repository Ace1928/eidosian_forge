import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_noTimeout(self):
    """
        Check that receiving data is delaying the timeout of the connection.
        """
    self.proto.makeConnection(StringTransport())
    self.clock.pump([0, 0.5, 1.0, 1.0])
    self.assertFalse(self.proto.timedOut)
    self.proto.dataReceived(b'hello there')
    self.clock.pump([0, 1.0, 1.0, 0.5])
    self.assertFalse(self.proto.timedOut)
    self.clock.pump([0, 1.0])
    self.assertTrue(self.proto.timedOut)