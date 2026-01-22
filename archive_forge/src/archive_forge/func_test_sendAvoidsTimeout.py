import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_sendAvoidsTimeout(self):
    """
        Make sure that writing data to a transport from a protocol
        constructed by a TimeoutFactory resets the timeout countdown.
        """
    self.clock.pump([0.0, 0.5, 1.0])
    self.assertFalse(self.wrappedProto.disconnected)
    self.proto.write(b'bytes bytes bytes')
    self.clock.pump([0.0, 1.0, 1.0])
    self.assertFalse(self.wrappedProto.disconnected)
    self.proto.writeSequence([b'bytes'] * 3)
    self.clock.pump([0.0, 1.0, 1.0])
    self.assertFalse(self.wrappedProto.disconnected)
    self.clock.pump([0.0, 2.0])
    self.assertTrue(self.wrappedProto.disconnected)