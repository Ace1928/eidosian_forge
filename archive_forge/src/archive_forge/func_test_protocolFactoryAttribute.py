import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_protocolFactoryAttribute(self):
    """
        Make sure protocol.factory is the wrapped factory, not the wrapping
        factory.
        """
    f = Server()
    wf = policies.WrappingFactory(f)
    p = wf.buildProtocol(address.IPv4Address('TCP', '127.0.0.1', 35))
    self.assertIs(p.wrappedProtocol.factory, f)