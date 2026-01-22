import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_breakReferenceCycle(self):
    """
        L{policies.ProtocolWrapper.connectionLost} sets C{wrappedProtocol} to
        C{None} in order to break reference cycle between wrapper and wrapped
        protocols.
        :return:
        """
    wrapper = policies.ProtocolWrapper(policies.WrappingFactory(Server()), protocol.Protocol())
    transport = StringTransportWithDisconnection()
    transport.protocol = wrapper
    wrapper.makeConnection(transport)
    self.assertIsNotNone(wrapper.wrappedProtocol)
    transport.loseConnection()
    self.assertIsNone(wrapper.wrappedProtocol)