import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_protocolLogPrefix(self):
    """
        L{ProtocolWrapper.logPrefix} is customized to mention both the original
        protocol and the wrapper.
        """
    server = Server()
    factory = policies.WrappingFactory(server)
    protocol = factory.buildProtocol(address.IPv4Address('TCP', '127.0.0.1', 35))
    self.assertEqual('EchoProtocol (ProtocolWrapper)', protocol.logPrefix())